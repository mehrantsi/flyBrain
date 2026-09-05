use flybrain_engine::behavior::BehaviorMode;
use flybrain_engine::flight_behavior::FlightMode;
use flybrain_engine::foraging::ForagingMode;
use flybrain_engine::retina::RetinaSummary;
use flybrain_engine::world::DEFAULT_ASSETS_DIR;
use flybrain_engine::world_sim::SimulationStepper;
use std::time::Instant;

const V783_PACK: &str = "outputs/packs/flywire_v783";

#[test]
#[ignore = "requires the local v783 pack, MuJoCo, and an Apple Metal device"]
fn v783_multisensory_world_advances_the_complete_model_pack() {
    let mut simulation =
        SimulationStepper::new(DEFAULT_ASSETS_DIR, Some(V783_PACK), 500.0, 0.0).unwrap();
    simulation.place_food_ahead(2.5).unwrap();
    simulation.toggle_flight();
    simulation.set_brain_telemetry_enabled(true);
    simulation
        .set_retina_summaries([
            RetinaSummary {
                mean_intensity: 0.65,
                spatial_contrast: 0.2,
            },
            RetinaSummary {
                mean_intensity: 0.35,
                spatial_contrast: 0.3,
            },
        ])
        .unwrap();

    assert_eq!(simulation.brain_neuron_count(), 138_639);
    assert!(simulation.full_neural_io_enabled());
    let io = simulation.neural_io_stats();
    assert_eq!(io.groups, 46);
    assert_eq!(io.selected_root_ids, 13_656);
    assert_eq!(io.present_root_ids, 13_115);
    assert_eq!(io.missing_root_ids, 541);

    let mut totals = [0_u64; 4];
    let mut peak_flight_drive = 0.0_f64;
    let mut peak_flight_dn_rate = 0.0_f64;
    let mut takeoff_drive_windows = 0_usize;
    let mut consecutive_takeoff_drive_windows = 0_usize;
    let mut longest_takeoff_drive_windows = 0_usize;
    for _ in 0..250 {
        let snapshot = simulation.step_window().unwrap();
        totals[0] += snapshot.taste_event_delta;
        totals[1] += snapshot.olfactory_event_delta;
        totals[2] += snapshot.visual_event_delta;
        totals[3] += snapshot.flight_state_event_delta;
        peak_flight_drive = peak_flight_drive.max(snapshot.brain_flight_drive);
        takeoff_drive_windows += usize::from(
            snapshot.brain_flight_drive
                >= flybrain_engine::flight_behavior::TAKEOFF_DRIVE_THRESHOLD,
        );
        if snapshot.brain_flight_drive >= flybrain_engine::flight_behavior::TAKEOFF_DRIVE_THRESHOLD
        {
            consecutive_takeoff_drive_windows += 1;
            longest_takeoff_drive_windows =
                longest_takeoff_drive_windows.max(consecutive_takeoff_drive_windows);
        } else {
            consecutive_takeoff_drive_windows = 0;
        }
        peak_flight_dn_rate = peak_flight_dn_rate
            .max(snapshot.flight_dn_left_rate_hz)
            .max(snapshot.flight_dn_right_rate_hz);
    }

    let snapshot = simulation.snapshot();
    eprintln!(
        "t={:.3}s events={totals:?} population={:.1}Hz flight_dn_peak={peak_flight_dn_rate:.1}Hz flight_drive_peak={peak_flight_drive:.3} takeoff_drive_windows={takeoff_drive_windows} longest={longest_takeoff_drive_windows}",
        snapshot.time_seconds, snapshot.filtered_population_rate_hz,
    );
    assert!((snapshot.time_seconds - 0.5).abs() < 1e-9);
    let spontaneous_olfactory_events = 2_090_u64 * 8 / 2;
    assert!(
        totals[1] > spontaneous_olfactory_events,
        "food odor did not evoke events above the 8 Hz ORN baseline"
    );
    assert!(totals[2] > 0, "visual channels received no events");
    assert!(snapshot.filtered_population_rate_hz > 0.0);
    assert!(snapshot.cumulative_spiking_neuron_count > 0);
    assert!(snapshot.brain_field_sample_sequence >= 49);
    assert!(snapshot.brain_field_potential_mv.is_finite());
    assert!(snapshot.brain_field_dominant_frequency_hz.is_finite());
}

#[test]
#[ignore = "requires the local v783 pack, MuJoCo, and an Apple Metal device"]
fn v783_maintained_physical_sugar_contact_drives_taste_and_mn9() {
    let mut simulation =
        SimulationStepper::new(DEFAULT_ASSETS_DIR, Some(V783_PACK), 500.0, 0.0).unwrap();
    simulation.place_food_at_mouth().unwrap();

    let mut saw_taste = false;
    let mut saw_feed = false;
    let mut saw_post_meal = false;
    let mut saw_post_meal_release = false;
    let mut saw_neural_takeoff_after_meal = false;
    let mut minimum_post_meal_up_z = 1.0_f64;
    let mut mn9_spikes = 0_u64;
    for step in 0..6_000 {
        if !saw_post_meal {
            simulation.place_food_at_mouth().unwrap();
        }
        let snapshot = simulation.step_window().unwrap();
        saw_taste |= snapshot.taste_active;
        saw_feed |= snapshot.behavior_mode == BehaviorMode::Feed
            && snapshot.foraging_mode == ForagingMode::Feed;
        saw_post_meal |= snapshot.behavior_mode == BehaviorMode::DepartFood
            && snapshot.foraging_mode == ForagingMode::PostMeal;
        saw_post_meal_release |=
            saw_post_meal && snapshot.behavior_mode != BehaviorMode::DepartFood;
        if saw_post_meal {
            let [_, x, y, _] = simulation.world().root_quaternion();
            minimum_post_meal_up_z = minimum_post_meal_up_z.min(1.0 - 2.0 * (x * x + y * y));
        }
        saw_neural_takeoff_after_meal |= saw_post_meal_release
            && snapshot.flight_mode != FlightMode::Grounded
            && snapshot.brain_flight_drive
                >= flybrain_engine::flight_behavior::TAKEOFF_DRIVE_THRESHOLD;
        mn9_spikes += u64::from(snapshot.mn9_spike_delta);
        if (step + 1) % 500 == 0 {
            eprintln!(
                "sample t={:.3} flight={} forage={} behavior={} pos={:?} food={:.2} odor={:.3}/{:.3} taste={} mn9={} extension={:.2} flight-drive={:.3} flight-dn={:.1}/{:.1}",
                snapshot.time_seconds,
                snapshot.flight_mode.label(),
                snapshot.foraging_mode.label(),
                snapshot.behavior_mode.label(),
                snapshot.root_position,
                snapshot.food_distance,
                snapshot.odor_left,
                snapshot.odor_right,
                snapshot.taste_active,
                mn9_spikes,
                snapshot.feeding_extension,
                snapshot.brain_flight_drive,
                snapshot.flight_dn_left_rate_hz,
                snapshot.flight_dn_right_rate_hz,
            );
        }
        if saw_neural_takeoff_after_meal {
            break;
        }
    }

    assert!(saw_taste);
    assert!(saw_feed);
    assert!(mn9_spikes > 0);
    assert!(saw_post_meal);
    assert!(saw_post_meal_release);
    assert!(saw_neural_takeoff_after_meal);
    assert!(minimum_post_meal_up_z > 0.5);
}

#[test]
#[ignore = "requires the local v783 pack, MuJoCo, and an Apple Metal device"]
fn v783_descending_activity_drives_high_flight_without_spurious_landing() {
    let mut simulation =
        SimulationStepper::new(DEFAULT_ASSETS_DIR, Some(V783_PACK), 500.0, 0.0).unwrap();
    simulation.toggle_food().unwrap();
    simulation
        .set_retina_summaries([
            RetinaSummary {
                mean_intensity: 0.65,
                spatial_contrast: 0.2,
            },
            RetinaSummary {
                mean_intensity: 0.35,
                spatial_contrast: 0.3,
            },
        ])
        .unwrap();

    let mut first_airborne_time = None;
    let mut minimum_height = f64::INFINITY;
    let mut maximum_height = f64::NEG_INFINITY;
    let mut peak_drive = 0.0_f64;
    let mut maximum_abs_x = 0.0_f64;
    let mut maximum_abs_y = 0.0_f64;
    let mut flight_lift_sum = 0.0_f64;
    let mut flight_windows = 0_usize;
    let mut grounded_windows = 0_usize;
    let mut takeoff_windows = 0_usize;
    let mut cruise_windows = 0_usize;
    let mut landing_windows = 0_usize;
    let mut peak_frequency_scale = 0.0_f64;
    let mut minimum_altitude_target = f64::INFINITY;
    let mut maximum_altitude_target = f64::NEG_INFINITY;
    let mut minimum_altitude_control = f64::INFINITY;
    let mut maximum_altitude_control = f64::NEG_INFINITY;
    let mut peak_abs_altitude_control = 0.0_f64;
    let mut peak_flight_power_increase_rate_hz = 0.0_f64;
    let mut peak_flight_power_decrease_rate_hz = 0.0_f64;
    let mut flight_state_events = 0_u64;
    let mut saw_airborne = false;
    let mut saw_landing = false;
    let mut saw_grounded_after_flight = false;
    let mut first_landing = None;
    let started = Instant::now();
    let mut brain_wall_seconds = 0.0;
    let mut brain_encoding_seconds = 0.0;
    let mut brain_engine_seconds = 0.0;
    for step in 0..5_000 {
        let snapshot = simulation.step_window().unwrap();
        assert_eq!(snapshot.flight_odor_steering, 0.0);
        assert_eq!(snapshot.flight_wander_steering, 0.0);
        if !snapshot.taste_active {
            assert_eq!(
                snapshot.flight_brain_steering_contribution,
                snapshot.brain_flight_steering
            );
        }
        assert_eq!(snapshot.optic_flow_altitude_contribution_mm_s, 0.0);
        brain_wall_seconds += snapshot.brain_wall_seconds;
        brain_encoding_seconds += snapshot.brain_encoding_seconds;
        brain_engine_seconds += snapshot.brain_engine_seconds;
        flight_state_events += snapshot.flight_state_event_delta;
        peak_drive = peak_drive.max(snapshot.brain_flight_drive);
        minimum_height = minimum_height.min(snapshot.root_position[2]);
        maximum_height = maximum_height.max(snapshot.root_position[2]);
        maximum_abs_x = maximum_abs_x.max(snapshot.root_position[0].abs());
        maximum_abs_y = maximum_abs_y.max(snapshot.root_position[1].abs());
        peak_abs_altitude_control =
            peak_abs_altitude_control.max(snapshot.brain_altitude_control.abs());
        minimum_altitude_control = minimum_altitude_control.min(snapshot.brain_altitude_control);
        maximum_altitude_control = maximum_altitude_control.max(snapshot.brain_altitude_control);
        peak_flight_power_increase_rate_hz =
            peak_flight_power_increase_rate_hz.max(snapshot.flight_power_increase_rate_hz);
        peak_flight_power_decrease_rate_hz =
            peak_flight_power_decrease_rate_hz.max(snapshot.flight_power_decrease_rate_hz);
        if snapshot.flight_mode != FlightMode::Grounded && first_airborne_time.is_none() {
            first_airborne_time = Some(snapshot.time_seconds);
        }
        if snapshot.flight_mode != FlightMode::Grounded {
            saw_airborne = true;
        }
        if snapshot.flight_mode == FlightMode::Landing && !saw_landing {
            first_landing = Some(snapshot);
        }
        saw_landing |= snapshot.flight_mode == FlightMode::Landing;
        saw_grounded_after_flight |= saw_airborne && snapshot.flight_mode == FlightMode::Grounded;
        match snapshot.flight_mode {
            FlightMode::Grounded => grounded_windows += 1,
            FlightMode::Takeoff => takeoff_windows += 1,
            FlightMode::Cruise => cruise_windows += 1,
            FlightMode::Landing => landing_windows += 1,
        }
        if snapshot.flight_mode != FlightMode::Grounded {
            flight_lift_sum += snapshot.flight_vertical_force_to_weight;
            flight_windows += 1;
            peak_frequency_scale = peak_frequency_scale.max(snapshot.flight_frequency_scale);
            if snapshot.flight_mode == FlightMode::Cruise {
                minimum_altitude_target =
                    minimum_altitude_target.min(snapshot.flight_target_height_mm);
                maximum_altitude_target =
                    maximum_altitude_target.max(snapshot.flight_target_height_mm);
                assert!(snapshot.flight_target_height_mm >= snapshot.flight_altitude_bounds_mm[0]);
                assert!(snapshot.flight_target_height_mm <= snapshot.flight_altitude_bounds_mm[1]);
            }
        }
        if (step + 1) % 1_000 == 0 {
            eprintln!(
                "sample t={:.3} mode={} z={:.3} target={:.3} altitude={:+.3} DNg02={:.1} DNg07={:.1} amp={:.3} freq={:.3} steer={:.3} avoid={:.3} lift={:.3} contacts={}",
                snapshot.time_seconds,
                snapshot.flight_mode.label(),
                snapshot.root_position[2],
                snapshot.flight_target_height_mm,
                snapshot.brain_altitude_control,
                snapshot.flight_power_increase_rate_hz,
                snapshot.flight_power_decrease_rate_hz,
                snapshot.flight_amplitude_scale,
                snapshot.flight_frequency_scale,
                snapshot.flight_steering,
                snapshot.flight_boundary_avoidance,
                snapshot.flight_vertical_force_to_weight,
                snapshot.contact_count,
            );
        }
    }
    let snapshot = simulation.snapshot();
    eprintln!(
        "flight first={first_airborne_time:?} duty grounded/takeoff/cruise/landing={grounded_windows}/{takeoff_windows}/{cruise_windows}/{landing_windows} airborne={:.3} z=[{minimum_height:.3},{maximum_height:.3}] target=[{minimum_altitude_target:.3},{maximum_altitude_target:.3}] altitude_command=[{minimum_altitude_control:.3},{maximum_altitude_control:.3}] DNg02/07 peaks={peak_flight_power_increase_rate_hz:.1}/{peak_flight_power_decrease_rate_hz:.1} flight_events={flight_state_events} max_xy=[{maximum_abs_x:.3},{maximum_abs_y:.3}] final={:?} quat={:?} mode={} contacts={} peak_drive={peak_drive:.3} mean_lift={:.3} peak_freq={peak_frequency_scale:.3} first_landing={first_landing:?} encode={brain_encoding_seconds:.3}s metal={brain_engine_seconds:.3}s bridge={brain_wall_seconds:.3}s total={:.3}s",
        flight_windows as f64 / 5_000.0,
        snapshot.root_position,
        simulation.world().root_quaternion(),
        snapshot.flight_mode.label(),
        snapshot.contact_count,
        flight_lift_sum / flight_windows.max(1) as f64,
        started.elapsed().as_secs_f64(),
    );
    assert!(first_airborne_time.is_some());
    assert!(flight_windows as f64 / 5_000.0 > 0.1);
    assert!(maximum_height > 100.0);
    assert!(flight_state_events > 0);
    assert!(peak_drive > flybrain_engine::flight_behavior::TAKEOFF_DRIVE_THRESHOLD);
    assert!(maximum_altitude_target > 100.0);
    assert!(maximum_altitude_target - minimum_altitude_target > 15.0);
    assert!(maximum_abs_x < 299.0);
    assert!(maximum_abs_y < 219.0);
    assert!(!saw_landing);
    assert!(!saw_grounded_after_flight);
    assert!(snapshot.contact_count <= 6);
    assert!(snapshot.root_position.iter().all(|value| value.is_finite()));
}
