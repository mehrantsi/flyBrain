#![cfg(target_os = "macos")]

use flybrain_engine::brain_bridge::BrainBodyBridge;
use flybrain_engine::embodiment::SensorySample;
use flybrain_engine::olfaction::OlfactoryTransducer;
use flybrain_engine::pack::ConnectomePack;

#[test]
#[ignore = "requires the MaleCNS pack and Apple Metal"]
fn matched_orn_readout_resolves_swapped_food_odor() {
    let pack = ConnectomePack::open("outputs/packs/male_cns_v1").unwrap();
    let mut contrasts = Vec::new();
    for (name, factors) in [
        ("none", [0.0, 0.0]),
        ("balanced", [1.0, 1.0]),
        ("left", [1.1, 0.9]),
        ("right", [0.9, 1.1]),
        ("weak_left", [1.003, 0.997]),
        ("weak_right", [0.997, 1.003]),
    ] {
        let mut bridge = BrainBodyBridge::new_with_neural_io(
            &pack,
            "assets/neuromechfly/male_cns_v1_neural_io.json",
        )
        .unwrap();
        let sample = SensorySample {
            food_odor_activation: factors.map(|factor| [0.08 * factor, 0.02 * factor, 0.0, 0.0]),
            ..SensorySample::default()
        };
        let mut mean = [0.0; 2];
        let mut contrast = 0.0;
        for step in 0..1000 {
            let readout = bridge
                .run_window(&sample, 20)
                .unwrap()
                .cns_olfactory
                .unwrap();
            if step >= 500 {
                for (total, rate) in mean.iter_mut().zip(readout.rate_hz) {
                    *total += rate / 500.0;
                }
                contrast += readout.contrast / 500.0;
            }
        }
        eprintln!("{name}: rate_hz={mean:?}, contrast={contrast}");
        contrasts.push(contrast);
    }
    assert!(contrasts[2] > contrasts[1]);
    assert!(contrasts[3] < contrasts[1]);
    assert!(contrasts[4] > contrasts[5]);
}

#[test]
#[ignore = "requires the MaleCNS pack and Apple Metal"]
fn whole_cns_olfaction_concentration_readout_tracks_balanced_input() {
    const BRAIN_STEPS: usize = 20;
    const WINDOW_SECONDS: f64 = 0.002;
    const TOTAL_WINDOWS: usize = 1_000;
    const MEASURE_FROM_WINDOW: usize = 750;
    const MEASURE_WINDOWS: f64 = (TOTAL_WINDOWS - MEASURE_FROM_WINDOW) as f64;
    let pack = ConnectomePack::open("outputs/packs/male_cns_v1").unwrap();
    let mut measurements = Vec::new();
    for concentration_ppm in [0.2, 1.0, 4.0, 12.0] {
        let mut bridge = BrainBodyBridge::new_with_neural_io(
            &pack,
            "assets/neuromechfly/male_cns_v1_neural_io.json",
        )
        .unwrap();
        let mut transducer = OlfactoryTransducer::default();
        let mut concentration_sum = [0.0; 2];
        let mut band_sum = [[0.0; 2]; 2];
        for step in 0..TOTAL_WINDOWS {
            let olfactory = transducer
                .update([concentration_ppm; 2], WINDOW_SECONDS)
                .unwrap();
            let sample = SensorySample {
                odor_intensity: 0.5
                    * (olfactory.perceived_intensity[0] + olfactory.perceived_intensity[1]),
                odor_left: olfactory.perceived_intensity[0],
                odor_right: olfactory.perceived_intensity[1],
                food_odor_activation: olfactory.receptor_activation,
                ..SensorySample::default()
            };
            let readout = bridge
                .run_window(&sample, BRAIN_STEPS)
                .unwrap()
                .cns_olfactory
                .unwrap();
            if step >= MEASURE_FROM_WINDOW {
                for (total, value) in concentration_sum.iter_mut().zip(readout.concentration_ppm) {
                    *total += value / MEASURE_WINDOWS;
                }
                for (band_total, band_values) in band_sum.iter_mut().zip(readout.band_rate_hz) {
                    for (total, value) in band_total.iter_mut().zip(band_values) {
                        *total += value / MEASURE_WINDOWS;
                    }
                }
            }
        }
        eprintln!(
            "input_concentration_ppm={concentration_ppm:.1} decoded_concentration_ppm={concentration_sum:?} band_rate_hz={band_sum:?}"
        );
        measurements.push((concentration_ppm, concentration_sum, band_sum));
    }

    for side in 0..2 {
        for pair in measurements.windows(2) {
            assert!(
                pair[1].1[side] >= pair[0].1[side],
                "decoded concentration is not monotonic on side {side}: {:?}",
                measurements
            );
        }
        assert!(
            measurements[0].1[side] < 2.0,
            "low concentration decoded too high"
        );
        assert!(
            measurements[3].1[side] > 6.0,
            "high concentration decoded too low"
        );
    }
}
