use flybrain_engine::brain_bridge::BrainBodyBridge;
use flybrain_engine::embodiment::SensorySample;
use flybrain_engine::pack::ConnectomePack;

const V783_PACK: &str = "outputs/packs/flywire_v783";
const NEURAL_IO: &str = "assets/neuromechfly/flywire_v783_neural_io.json";

#[test]
#[ignore = "requires the local v783 pack and an Apple Metal device"]
fn report_altitude_candidate_responses_to_available_sensory_classes() {
    let pack = ConnectomePack::open(V783_PACK).unwrap();
    let scenarios = [
        ("baseline", SensorySample::default()),
        (
            "visual",
            SensorySample {
                visual_left: 0.8,
                visual_right: 0.8,
                visual_contrast_left: 0.8,
                visual_contrast_right: 0.8,
                ..SensorySample::default()
            },
        ),
        (
            "odor",
            SensorySample {
                odor_intensity: 0.8,
                odor_left: 0.8,
                odor_right: 0.8,
                food_odor_activation: [[0.8; 4]; 2],
                ..SensorySample::default()
            },
        ),
        (
            "flight-state",
            SensorySample {
                flight_angular_speed_rad_s: 120.0,
                ..SensorySample::default()
            },
        ),
        (
            "jo-e",
            SensorySample {
                flight_mechanosensory: 0.8,
                ..SensorySample::default()
            },
        ),
        (
            "taste",
            SensorySample {
                taste_valence: 1.0,
                ..SensorySample::default()
            },
        ),
        (
            "combined-flight",
            SensorySample {
                visual_left: 0.8,
                visual_right: 0.8,
                visual_contrast_left: 0.8,
                visual_contrast_right: 0.8,
                flight_angular_speed_rad_s: 120.0,
                flight_mechanosensory: 0.8,
                ..SensorySample::default()
            },
        ),
    ];

    for (name, sample) in scenarios {
        let mut bridge = BrainBodyBridge::new_with_neural_io(&pack, NEURAL_IO).unwrap();
        let response = bridge.run_window(&sample, 10_000).unwrap();
        eprintln!(
            "{name}: DNg02={:.3}Hz DNg07={:.3}Hz altitude={:+.4} landing={:.3}Hz flight={:.3}/{:.3}Hz walking={:.3}/{:.3}Hz",
            response.flight_power_increase_rate_hz,
            response.flight_power_decrease_rate_hz,
            response.brain_altitude_control,
            response.landing_dn_rate_hz,
            response.flight_left_rate_hz,
            response.flight_right_rate_hz,
            response.walking_left_rate_hz,
            response.walking_right_rate_hz,
        );
    }
}
