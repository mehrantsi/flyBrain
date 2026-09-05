#![cfg(target_os = "macos")]

use flybrain_engine::flight::{
    FlightCommand, FlightDynamicsParameters, FlightRuntime, FlightStabilizer,
};
use flybrain_engine::gait::GaitLibrary;
use flybrain_engine::world::{JOINT_ACTUATOR_COUNT, MuJoCoWorld};

const ASSETS_DIR: &str = "assets/neuromechfly";
const LANDING_HEIGHT_MM: f64 = 0.6;

struct AuthorityResult {
    initial_position: [f64; 3],
    touchdown_position: [f64; 3],
    final_position: [f64; 3],
    maximum_horizontal_speed: f64,
    maximum_vertical_speed: f64,
    touchdown_up_z: f64,
    minimum_up_z: f64,
}

fn up_axis_z(quaternion: [f64; 4]) -> f64 {
    let [_, x, y, _] = quaternion;
    1.0 - 2.0 * (x * x + y * y)
}

fn run_authority(
    velocity_gain_per_s: f64,
    maximum_horizontal_force_weight: f64,
) -> AuthorityResult {
    let mut world = MuJoCoWorld::from_assets_dir(ASSETS_DIR).unwrap();
    let dynamics = FlightDynamicsParameters {
        velocity_gain_per_s,
        maximum_horizontal_force_weight,
        ..Default::default()
    };
    let half_pitch = 0.5 * dynamics.target_pitch_rad;
    world.data_mut().qpos_mut()[2] = 8.0;
    world.data_mut().qpos_mut()[3..7].copy_from_slice(&[
        half_pitch.cos(),
        0.0,
        half_pitch.sin(),
        0.0,
    ]);
    world.data_mut().qvel_mut()[..6].fill(0.0);
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
    let mut maximum_horizontal_speed = 0.0_f64;
    let mut maximum_vertical_speed = 0.0_f64;
    let mut touchdown_position = None;
    for _ in 0..10_000 {
        world.set_joint_controls(&neutral_joint_controls).unwrap();
        world.set_adhesion_controls(&[0.0; 6]).unwrap();
        let velocity = world.root_velocity();
        let height_error = LANDING_HEIGHT_MM - world.root_position()[2];
        let frequency_scale =
            (1.25 + 0.004 * height_error - 0.0005 * velocity[5]).clamp(1.05, 1.45);
        let command = stabilizer
            .command_with_base_limited(
                world.root_quaternion(),
                velocity,
                FlightCommand {
                    enabled: true,
                    amplitude: 1.0,
                    horizontal_speed_scale: 0.0,
                    altitude_target_mm: Some(LANDING_HEIGHT_MM),
                    body_pitch_target_rad: Some(0.0),
                    frequency_scale,
                    ..FlightCommand::default()
                },
                landing_amplitude_scale,
                flight.config(),
            )
            .unwrap();
        flight
            .advance(&mut world, command, [35.0, 8.0, 0.0])
            .unwrap();
        let velocity = world.root_velocity();
        maximum_horizontal_speed = maximum_horizontal_speed.max(velocity[3].hypot(velocity[4]));
        maximum_vertical_speed = maximum_vertical_speed.max(velocity[5].abs());
        let contact_count = world
            .support_contacts()
            .unwrap()
            .into_iter()
            .filter(|&contact| contact)
            .count();
        if contact_count >= 2 {
            touchdown_position = Some(world.root_position());
            break;
        }
    }
    let touchdown_position = touchdown_position.unwrap_or_else(|| {
        panic!(
            "P={velocity_gain_per_s} cap={maximum_horizontal_force_weight} no touchdown t={} z={} vz={} max_hspeed={} max_vspeed={}",
            world.time(),
            world.root_position()[2],
            world.root_velocity()[5],
            maximum_horizontal_speed,
            maximum_vertical_speed,
        )
    });
    let touchdown_up_z = up_axis_z(world.root_quaternion());
    let mut phase_rad = 0.0;
    let mut minimum_up_z = touchdown_up_z;
    for _ in 0..20_000 {
        let gait_command = gait.sample(phase_rad, 1.0, 0.0).unwrap();
        world
            .set_joint_controls(&gait_command.joint_controls)
            .unwrap();
        world.set_adhesion_controls(&gait_command.adhesion).unwrap();
        world.set_feeding_extension(0.0).unwrap();
        flight
            .advance(&mut world, FlightCommand::default(), [0.0; 3])
            .unwrap();
        phase_rad = gait.advance_phase(phase_rad, timestep, 0.5).unwrap();
        minimum_up_z = minimum_up_z.min(up_axis_z(world.root_quaternion()));
    }
    AuthorityResult {
        initial_position,
        touchdown_position,
        final_position: world.root_position(),
        maximum_horizontal_speed,
        maximum_vertical_speed,
        touchdown_up_z,
        minimum_up_z,
    }
}

#[test]
#[ignore = "physical horizontal-authority bench"]
fn reports_landing_horizontal_authority_tradeoff() {
    for (velocity_gain_per_s, maximum_horizontal_force_weight) in [
        (20.0, 1.0),
        (50.0, 1.0),
        (100.0, 1.0),
        (100.0, 2.0),
        (200.0, 2.0),
    ] {
        let result = run_authority(velocity_gain_per_s, maximum_horizontal_force_weight);
        let touchdown_shift = (result.touchdown_position[0] - result.initial_position[0])
            .hypot(result.touchdown_position[1] - result.initial_position[1]);
        let walking_displacement = (result.final_position[0] - result.touchdown_position[0])
            .hypot(result.final_position[1] - result.touchdown_position[1]);
        eprintln!(
            "P={velocity_gain_per_s} cap={maximum_horizontal_force_weight} touchdown_shift={touchdown_shift} max_hspeed={} max_vspeed={} touchdown_up_z={} minimum_up_z={} walking_displacement={walking_displacement} touchdown={:?} final={:?}",
            result.maximum_horizontal_speed,
            result.maximum_vertical_speed,
            result.touchdown_up_z,
            result.minimum_up_z,
            result.touchdown_position,
            result.final_position,
        );
        assert!(result.touchdown_up_z > 0.5);
        assert!(result.minimum_up_z > 0.5);
    }
}
