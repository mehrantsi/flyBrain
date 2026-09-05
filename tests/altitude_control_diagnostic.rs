#![cfg(target_os = "macos")]

use flybrain_engine::flight::{
    FlightCommand, FlightDynamicsParameters, FlightRuntime, FlightStabilizer,
};
use flybrain_engine::world::MuJoCoWorld;

#[test]
fn altitude_control_rejects_steady_state_lift_bias() {
    for target_height_mm in [8.0, 20.0] {
        let mut world = MuJoCoWorld::from_assets_dir("assets/neuromechfly").unwrap();
        let dynamics = FlightDynamicsParameters::default();
        let half_pitch = dynamics.target_pitch_rad * 0.5;
        world.data_mut().qpos_mut()[2] = target_height_mm;
        world.data_mut().qpos_mut()[3..7].copy_from_slice(&[
            half_pitch.cos(),
            0.0,
            half_pitch.sin(),
            0.0,
        ]);
        world.data_mut().qvel_mut()[..6].fill(0.0);
        world.data_mut().forward();
        let mut flight =
            FlightRuntime::new_with_parameters("assets/neuromechfly", &world, dynamics).unwrap();
        let stabilizer = FlightStabilizer::from_dynamics(dynamics).unwrap();
        let base = FlightCommand {
            enabled: true,
            amplitude: 1.0,
            steering: 0.0,
            wing_steering_scale: 1.0,
            horizontal_speed_scale: 0.05,
            altitude_target_mm: Some(target_height_mm),
            frequency_scale: 1.25,
            ..Default::default()
        };
        let weight = world.model().body_mass().iter().sum::<f64>() * 9_810.0;
        let maximum_vertical_force = weight * dynamics.maximum_vertical_force_weight;
        let mut height_sum = 0.0;
        let mut vertical_velocity_sum = 0.0;
        let mut maximum_commanded_force = 0.0_f64;
        let samples = 10_000_u64;
        for step in 0..20_000_u64 {
            let command = stabilizer
                .command_with_base_limited(
                    world.root_quaternion(),
                    world.root_velocity(),
                    base,
                    1.0,
                    flight.config(),
                )
                .unwrap();
            let telemetry = flight
                .advance(&mut world, command, [35.0, 8.0, 0.0])
                .unwrap();
            if step >= 10_000 {
                let height = world.root_position()[2];
                let vertical_velocity = world.root_velocity()[5];
                let commanded_force = telemetry.engineered_body_velocity_force_g_mm_s2.z();
                height_sum += height;
                vertical_velocity_sum += vertical_velocity;
                maximum_commanded_force = maximum_commanded_force.max(commanded_force.abs());
            }
        }
        let count = samples as f64;
        let mean_height = height_sum / count;
        let mean_vertical_velocity = vertical_velocity_sum / count;
        assert!(
            (mean_height - target_height_mm).abs() < 0.1,
            "target={target_height_mm} mean_height={mean_height}"
        );
        assert!(
            mean_vertical_velocity.abs() < 0.1,
            "target={target_height_mm} mean_vz={mean_vertical_velocity}"
        );
        assert!(
            maximum_commanded_force <= maximum_vertical_force + 1e-9,
            "target={target_height_mm} max_commanded_force={maximum_commanded_force} limit={maximum_vertical_force}"
        );
    }
}
