#![cfg(target_os = "macos")]

use flybrain_engine::flight::{
    FlightCommand, FlightDynamicsParameters, FlightRuntime, FlightStabilizer,
};
use flybrain_engine::world::MuJoCoWorld;

struct Tracking {
    mean_yaw_rate: f64,
    mean_forward_speed: f64,
    maximum_roll: f64,
    maximum_pitch_error: f64,
    final_heading: [f64; 2],
}

fn track(base: FlightCommand, seconds: f64, initial_speed: f64) -> Tracking {
    let mut world = MuJoCoWorld::from_assets_dir("assets/neuromechfly").unwrap();
    let dynamics = FlightDynamicsParameters::default();
    let half_pitch = dynamics.target_pitch_rad * 0.5;
    world.data_mut().qpos_mut()[2] = 20.0;
    world.data_mut().qpos_mut()[3..7].copy_from_slice(&[
        half_pitch.cos(),
        0.0,
        half_pitch.sin(),
        0.0,
    ]);
    world.data_mut().qvel_mut()[..6].fill(0.0);
    world.data_mut().qvel_mut()[0] = initial_speed;
    world.data_mut().forward();
    let mut flight =
        FlightRuntime::new_with_parameters("assets/neuromechfly", &world, dynamics).unwrap();
    let stabilizer = FlightStabilizer::from_dynamics(dynamics).unwrap();
    let mut result = Tracking {
        mean_yaw_rate: 0.0,
        mean_forward_speed: 0.0,
        maximum_roll: 0.0,
        maximum_pitch_error: 0.0,
        final_heading: [0.0; 2],
    };
    let mut samples = 0;
    while world.time() < seconds {
        let command = stabilizer
            .command_with_base_limited(
                world.root_quaternion(),
                world.root_velocity(),
                base,
                1.0,
                flight.config(),
            )
            .unwrap();
        flight
            .advance(&mut world, command, [35.0, 8.0, 0.0])
            .unwrap();
        let [w, x, y, z] = world.root_quaternion();
        let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
        let roll = (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y));
        let pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin();
        result.maximum_roll = result.maximum_roll.max(roll.abs());
        result.maximum_pitch_error = result
            .maximum_pitch_error
            .max((pitch - dynamics.target_pitch_rad).abs());
        result.final_heading = [yaw.cos(), yaw.sin()];
        if world.time() > seconds - 1.0 {
            let velocity = world.root_velocity();
            result.mean_yaw_rate += velocity[2];
            result.mean_forward_speed += velocity[3] * yaw.cos() + velocity[4] * yaw.sin();
            samples += 1;
        }
    }
    result.mean_yaw_rate /= samples as f64;
    result.mean_forward_speed /= samples as f64;
    result
}

fn flight_command() -> FlightCommand {
    FlightCommand {
        enabled: true,
        amplitude: 1.0,
        horizontal_speed_scale: 0.05,
        altitude_target_mm: Some(20.0),
        frequency_scale: 1.25,
        ..Default::default()
    }
}

#[test]
fn physical_flight_tracks_neutral_and_signed_neural_steering() {
    for steering in [0.0, 0.25, -0.25] {
        let result = track(
            FlightCommand {
                steering,
                ..flight_command()
            },
            2.0,
            0.0,
        );
        assert!(
            (result.mean_yaw_rate - 2.0 * steering).abs() < 0.07,
            "steering={steering} yaw_rate={}",
            result.mean_yaw_rate
        );
        assert!(
            (result.mean_forward_speed - 15.0).abs() < 0.5,
            "forward speed={}",
            result.mean_forward_speed
        );
        assert!(result.maximum_roll < 0.25);
        assert!(result.maximum_pitch_error < 0.25);
    }
}

#[test]
fn stopping_for_landing_rejects_horizontal_drift() {
    let result = track(
        FlightCommand {
            horizontal_speed_scale: 0.0,
            ..flight_command()
        },
        2.0,
        100.0,
    );
    assert!(
        result.mean_forward_speed.abs() < 0.5,
        "stopped forward speed={}",
        result.mean_forward_speed
    );
}

#[test]
fn wall_heading_half_turn_preserves_attitude_and_speed() {
    let result = track(
        FlightCommand {
            heading_target_xy: Some([-1.0, 0.0]),
            wing_steering_scale: 0.0,
            ..flight_command()
        },
        4.0,
        0.0,
    );
    assert!(
        result.final_heading[0] < -0.99,
        "heading={:?}",
        result.final_heading
    );
    assert!(
        result.mean_yaw_rate.abs() < 0.1,
        "yaw rate={}",
        result.mean_yaw_rate
    );
    assert!(
        (result.mean_forward_speed - 15.0).abs() < 0.5,
        "forward speed={}",
        result.mean_forward_speed
    );
    assert!(result.maximum_roll < 0.25, "roll={}", result.maximum_roll);
    assert!(
        result.maximum_pitch_error < 0.25,
        "pitch error={}",
        result.maximum_pitch_error
    );
}
