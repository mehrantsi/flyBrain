#![cfg(target_os = "macos")]

use flybrain_engine::flight::{FlightCommand, FlightRuntime};
use flybrain_engine::gait::{GaitCommand, GaitLibrary};
use flybrain_engine::world::{JOINT_ACTUATOR_COUNT, MuJoCoWorld};

const ASSETS_DIR: &str = "assets/neuromechfly";
const CONTROL_HZ: f64 = 500.0;
const SETTLE_SECONDS: f64 = 0.5;
const MEASURE_SECONDS: f64 = 4.0;
const GAIT_SPEED_GAIN: f64 = 0.5;

struct WalkingResult {
    signed_yaw_change_rad: f64,
    signed_yaw_rate_rad_s: f64,
    planar_displacement_mm: f64,
    maximum_planar_speed_mm_s: f64,
    minimum_up_z: f64,
}

fn yaw_rad(quaternion: [f64; 4]) -> f64 {
    let [w, x, y, z] = quaternion;
    (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z))
}

fn wrapped_angle_delta(current: f64, previous: f64) -> f64 {
    (current - previous + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
        - std::f64::consts::PI
}

fn run_walking(steering: f64) -> WalkingResult {
    run_walking_with(|gait, phase_rad| gait.sample(phase_rad, 1.0, steering).unwrap())
}

fn run_bilateral(side_drive: [f64; 2]) -> WalkingResult {
    run_walking_with(|gait, phase_rad| gait.sample_bilateral(phase_rad, side_drive).unwrap())
}

fn run_walking_with(sample: impl Fn(&GaitLibrary, f64) -> GaitCommand) -> WalkingResult {
    let mut world = MuJoCoWorld::from_assets_dir(ASSETS_DIR).unwrap();
    let gait = GaitLibrary::open(format!("{ASSETS_DIR}/tripod_gait.json")).unwrap();
    let mut flight = FlightRuntime::new(ASSETS_DIR, &world).unwrap();
    let timestep = world.timestep_seconds();
    let control_steps = (1.0 / CONTROL_HZ / timestep).round() as usize;
    assert_eq!(control_steps, 20);
    let window_seconds = control_steps as f64 * timestep;
    let settle_windows = (SETTLE_SECONDS / window_seconds).round() as usize;
    let measure_windows = (MEASURE_SECONDS / window_seconds).round() as usize;
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
    let mut previous_yaw = yaw_rad(world.root_quaternion());
    let mut unwrapped_yaw_change_rad = 0.0;
    let mut measurement_start_position = None;
    let mut maximum_planar_speed_mm_s = 0.0_f64;
    let mut minimum_up_z = f64::INFINITY;
    for window in 0..(settle_windows + measure_windows) {
        let gait_command = sample(&gait, phase_rad);
        let ramp = if window < settle_windows {
            ((window as f64 * window_seconds) / SETTLE_SECONDS).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let joint_controls: [f64; JOINT_ACTUATOR_COUNT] = std::array::from_fn(|index| {
            neutral_joint_controls[index] * (1.0 - ramp) + gait_command.joint_controls[index] * ramp
        });
        let mut joint_controls = joint_controls;
        for (index, control) in joint_controls.iter_mut().enumerate() {
            let [minimum, maximum] = world.actuator_control_range(index).unwrap();
            *control = control.clamp(minimum, maximum);
        }
        let adhesion = if ramp >= 0.5 {
            gait_command.adhesion
        } else {
            [0.0; 6]
        };
        world.set_joint_controls(&joint_controls).unwrap();
        world.set_adhesion_controls(&adhesion).unwrap();
        if window == settle_windows {
            measurement_start_position = Some(world.root_position());
            previous_yaw = yaw_rad(world.root_quaternion());
        }
        for _ in 0..control_steps {
            flight
                .advance(&mut world, FlightCommand::default(), [0.0; 3])
                .unwrap();
            let current_yaw = yaw_rad(world.root_quaternion());
            if window >= settle_windows {
                unwrapped_yaw_change_rad += wrapped_angle_delta(current_yaw, previous_yaw);
                let velocity = world.root_velocity();
                maximum_planar_speed_mm_s =
                    maximum_planar_speed_mm_s.max(velocity[3].hypot(velocity[4]));
                let [_, x, y, _] = world.root_quaternion();
                minimum_up_z = minimum_up_z.min(1.0 - 2.0 * (x * x + y * y));
            }
            previous_yaw = current_yaw;
        }
        phase_rad = gait
            .advance_phase(phase_rad, window_seconds, GAIT_SPEED_GAIN)
            .unwrap();
    }
    let initial_position = measurement_start_position.unwrap();
    let final_position = world.root_position();
    let planar_displacement_mm =
        (final_position[0] - initial_position[0]).hypot(final_position[1] - initial_position[1]);
    WalkingResult {
        signed_yaw_change_rad: unwrapped_yaw_change_rad,
        signed_yaw_rate_rad_s: unwrapped_yaw_change_rad / MEASURE_SECONDS,
        planar_displacement_mm,
        maximum_planar_speed_mm_s,
        minimum_up_z,
    }
}

#[test]
fn gait_signed_steering_has_bounded_physical_turning() {
    let neutral = run_walking(0.0);
    let positive = run_walking(0.7);
    let negative = run_walking(-0.7);
    eprintln!(
        "neutral yaw={:.6} rad/{:.6} rad_s displacement={:.3} mm max_speed={:.3} min_up_z={:.3}; positive yaw={:.6} rad/{:.6} rad_s displacement={:.3} mm max_speed={:.3} min_up_z={:.3}; negative yaw={:.6} rad/{:.6} rad_s displacement={:.3} mm max_speed={:.3} min_up_z={:.3}",
        neutral.signed_yaw_change_rad,
        neutral.signed_yaw_rate_rad_s,
        neutral.planar_displacement_mm,
        neutral.maximum_planar_speed_mm_s,
        neutral.minimum_up_z,
        positive.signed_yaw_change_rad,
        positive.signed_yaw_rate_rad_s,
        positive.planar_displacement_mm,
        positive.maximum_planar_speed_mm_s,
        positive.minimum_up_z,
        negative.signed_yaw_change_rad,
        negative.signed_yaw_rate_rad_s,
        negative.planar_displacement_mm,
        negative.maximum_planar_speed_mm_s,
        negative.minimum_up_z,
    );
    assert!(neutral.signed_yaw_rate_rad_s.abs() < 0.05);
    assert!(positive.signed_yaw_rate_rad_s > 0.5);
    assert!(negative.signed_yaw_rate_rad_s < -0.5);
    assert!(neutral.minimum_up_z > 0.9);
    assert!(positive.minimum_up_z > 0.9);
    assert!(negative.minimum_up_z > 0.9);
}

#[test]
fn bilateral_gait_pivots_have_signed_yaw_and_stay_upright() {
    let left = run_bilateral([-0.7, 0.7]);
    let right = run_bilateral([0.7, -0.7]);
    let forward = run_bilateral([1.0, 1.0]);
    eprintln!(
        "pivot_left yaw={:.6} rad/{:.6} rad_s displacement={:.3} mm max_speed={:.3} min_up_z={:.3}; pivot_right yaw={:.6} rad/{:.6} rad_s displacement={:.3} mm max_speed={:.3} min_up_z={:.3}; forward yaw={:.6} rad/{:.6} rad_s displacement={:.3} mm max_speed={:.3} min_up_z={:.3}",
        left.signed_yaw_change_rad,
        left.signed_yaw_rate_rad_s,
        left.planar_displacement_mm,
        left.maximum_planar_speed_mm_s,
        left.minimum_up_z,
        right.signed_yaw_change_rad,
        right.signed_yaw_rate_rad_s,
        right.planar_displacement_mm,
        right.maximum_planar_speed_mm_s,
        right.minimum_up_z,
        forward.signed_yaw_change_rad,
        forward.signed_yaw_rate_rad_s,
        forward.planar_displacement_mm,
        forward.maximum_planar_speed_mm_s,
        forward.minimum_up_z,
    );
    assert!(left.signed_yaw_rate_rad_s > 0.35);
    assert!(right.signed_yaw_rate_rad_s < -0.35);
    assert!(left.minimum_up_z > 0.9);
    assert!(right.minimum_up_z > 0.9);
    assert!(forward.minimum_up_z > 0.9);
}
