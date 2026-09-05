use flybrain_engine::retina::RetinaSummary;
use flybrain_engine::world::DEFAULT_ASSETS_DIR;
use flybrain_engine::world_sim::SimulationStepper;

const V783_PACK: &str = "outputs/packs/flywire_v783";

#[test]
#[ignore = "requires the local v783 pack, MuJoCo, and an Apple Metal device"]
fn v783_airborne_odor_probe_reports_trajectory_and_plume() {
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

    let apple = [105.0, 70.0, 58.0];
    let initial_apple_distance = distance(simulation.snapshot().root_position, apple);
    let mut minimum_apple_distance = initial_apple_distance;
    let mut peak_odor = 0.0_f64;
    let mut peak_odor_asymmetry = 0.0_f64;
    let mut first_airborne = None;
    for step in 0..2_500 {
        let snapshot = simulation.step_window().unwrap();
        peak_odor = peak_odor.max(snapshot.odor_intensity);
        peak_odor_asymmetry =
            peak_odor_asymmetry.max((snapshot.odor_left - snapshot.odor_right).abs());
        minimum_apple_distance =
            minimum_apple_distance.min(distance(snapshot.root_position, apple));
        if snapshot.flight_mode != flybrain_engine::flight_behavior::FlightMode::Grounded
            && first_airborne.is_none()
        {
            first_airborne = Some(snapshot.time_seconds);
        }
        if (step + 1) % 500 == 0 {
            eprintln!(
                "t={:.1} mode={} pos={:?} apple_dist={:.1} odor=({:.4},{:.4}) drive={:.3} steer={:.3}",
                snapshot.time_seconds,
                snapshot.flight_mode.label(),
                snapshot.root_position,
                distance(snapshot.root_position, apple),
                snapshot.odor_left,
                snapshot.odor_right,
                snapshot.brain_flight_drive,
                snapshot.flight_steering,
            );
        }
    }
    let final_snapshot = simulation.snapshot();
    eprintln!(
        "initial_apple_distance={initial_apple_distance:.3} minimum_apple_distance={minimum_apple_distance:.3} final_apple_distance={:.3} first_airborne={first_airborne:?} final_pos={:?} peak_odor={peak_odor:.4} peak_asymmetry={peak_odor_asymmetry:.4}",
        distance(final_snapshot.root_position, apple),
        final_snapshot.root_position,
    );
    assert!(first_airborne.is_some());
    assert!(peak_odor.is_finite());
    assert!(peak_odor_asymmetry.is_finite());
    assert!(minimum_apple_distance < initial_apple_distance);
    let planar_wall_clearance = (300.0 - final_snapshot.root_position[0].abs())
        .min(220.0 - final_snapshot.root_position[1].abs());
    assert!(
        planar_wall_clearance > 5.0,
        "active flight remained pinned against a room wall"
    );
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
}
