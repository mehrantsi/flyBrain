use flybrain_engine::world::DEFAULT_ASSETS_DIR;
use flybrain_engine::world_sim::SimulationStepper;

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn front_tarsi_approach_the_antennae_during_manual_grooming() {
    let mut simulation =
        SimulationStepper::new(DEFAULT_ASSETS_DIR, None::<&str>, 500.0, 0.5).unwrap();
    simulation.toggle_flight();
    simulation.toggle_food().unwrap();
    for _ in 0..250 {
        simulation.step_window().unwrap();
    }
    let baseline = [
        distance(
            simulation.world().body_position("fly/lf_tarsus5").unwrap(),
            simulation.world().body_position("fly/l_funiculus").unwrap(),
        ),
        distance(
            simulation.world().body_position("fly/rf_tarsus5").unwrap(),
            simulation.world().body_position("fly/r_funiculus").unwrap(),
        ),
    ];

    simulation.request_grooming();
    let mut minimum = [f64::INFINITY; 2];
    let mut active_windows = 0;
    let mut maximum_contacts = 0;
    for _ in 0..1_000 {
        let snapshot = simulation.step_window().unwrap();
        maximum_contacts = maximum_contacts.max(snapshot.contact_count);
        if snapshot.grooming_active {
            active_windows += 1;
            minimum[0] = minimum[0].min(distance(
                simulation.world().body_position("fly/lf_tarsus5").unwrap(),
                simulation.world().body_position("fly/l_funiculus").unwrap(),
            ));
            minimum[1] = minimum[1].min(distance(
                simulation.world().body_position("fly/rf_tarsus5").unwrap(),
                simulation.world().body_position("fly/r_funiculus").unwrap(),
            ));
            assert_eq!(snapshot.grooming_support_leg_count, 4);
        }
    }

    eprintln!(
        "grooming tarsus-to-antenna distance baseline={baseline:?} minimum={minimum:?} active_windows={active_windows} max_contacts={maximum_contacts}"
    );
    assert!(active_windows > 500);
    assert!(minimum[0] < baseline[0]);
    assert!(minimum[1] < baseline[1]);
}
