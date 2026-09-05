use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::pack::ConnectomePack;
use crate::parameters::ModelParameters;
use crate::stimulus::EventSchedule;

#[derive(Clone, Debug, Deserialize)]
pub struct TickFixture {
    pub schema: String,
    pub schema_version: u32,
    pub case_id: String,
    pub parameters: ModelParameters,
    pub network: FixtureNetwork,
    pub initial_state: InitialState,
    pub overrides: Overrides,
    pub stimulus: Stimulus,
    pub run: RunConfiguration,
    pub acceptance: Acceptance,
    pub expected: Expected,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FixtureNetwork {
    pub neuron_count: usize,
    pub edge_count: usize,
    pub neuron_ids_u64: Vec<u64>,
    pub row_ptr_u32: Vec<u32>,
    pub destinations_u32: Vec<u32>,
    pub signed_counts_i16: Vec<i16>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct InitialState {
    pub v_mv: Vec<f64>,
    pub g_mv: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Overrides {
    pub silenced_sources: Vec<u32>,
    pub zero_refractory: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Stimulus {
    pub external_counts: Vec<ExternalCount>,
    pub source_spikes: Vec<SourceSpike>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct ExternalCount {
    pub tick: usize,
    pub neuron: u32,
    pub count: u8,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct SourceSpike {
    pub tick: usize,
    pub source: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RunConfiguration {
    pub steps: usize,
    pub record_slots: Vec<String>,
    pub state_layout: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Acceptance {
    pub state_abs_tol_mv: f64,
    pub time_abs_tol_ms: f64,
    pub spikes: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Expected {
    pub times_ms: Vec<f64>,
    pub spike_events: Vec<SpikeEvent>,
    pub v_end_mv: Vec<Vec<f64>>,
    pub g_end_mv: Vec<Vec<f64>>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub struct SpikeEvent {
    pub tick: usize,
    pub neuron: usize,
}

impl TickFixture {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes =
            fs::read(path).with_context(|| format!("reading fixture {}", path.display()))?;
        let fixture: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing fixture {}", path.display()))?;
        fixture.validate()?;
        Ok(fixture)
    }

    pub fn connectome(&self) -> Result<ConnectomePack> {
        ConnectomePack::from_arrays(
            &self.network.neuron_ids_u64,
            &self.network.row_ptr_u32,
            &self.network.destinations_u32,
            &self.network.signed_counts_i16,
        )
    }

    pub fn event_schedule(&self) -> Result<EventSchedule> {
        let targets: Vec<u32> = self
            .stimulus
            .external_counts
            .iter()
            .map(|event| event.neuron)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut counts = vec![0_u8; self.run.steps * targets.len()];
        for event in &self.stimulus.external_counts {
            let lane = targets.binary_search(&event.neuron).unwrap();
            counts[event.tick * targets.len() + lane] = event.count;
        }
        EventSchedule::new(targets, counts, self.run.steps, self.network.neuron_count)
    }

    fn validate(&self) -> Result<()> {
        if self.schema != "flybrain.tick-fixture" || self.schema_version != 1 {
            bail!("unsupported tick fixture schema or version");
        }
        self.parameters.validate()?;
        if self.network.neuron_ids_u64.len() != self.network.neuron_count
            || self.network.row_ptr_u32.len() != self.network.neuron_count + 1
            || self.network.destinations_u32.len() != self.network.edge_count
            || self.network.signed_counts_i16.len() != self.network.edge_count
        {
            bail!("fixture network dimensions are inconsistent");
        }
        self.connectome()?;
        if self.initial_state.v_mv.len() != self.network.neuron_count
            || self.initial_state.g_mv.len() != self.network.neuron_count
        {
            bail!("fixture initial state must have one value per neuron");
        }
        if self.run.record_slots != ["end"] || self.run.state_layout != "tick_major" {
            bail!("fixture must record tick-major end state");
        }
        if self.expected.times_ms.len() != self.run.steps
            || self.expected.v_end_mv.len() != self.run.steps
            || self.expected.g_end_mv.len() != self.run.steps
        {
            bail!("fixture expected output has the wrong number of steps");
        }
        if self
            .expected
            .v_end_mv
            .iter()
            .chain(&self.expected.g_end_mv)
            .any(|row| row.len() != self.network.neuron_count)
        {
            bail!("fixture expected state has the wrong neuron dimension");
        }
        if self.acceptance.spikes != "exact"
            || !self.acceptance.state_abs_tol_mv.is_finite()
            || self.acceptance.state_abs_tol_mv < 0.0
            || !self.acceptance.time_abs_tol_ms.is_finite()
            || self.acceptance.time_abs_tol_ms < 0.0
        {
            bail!("fixture acceptance policy is invalid");
        }
        if !self.stimulus.source_spikes.is_empty() {
            bail!("forced source-spike fixtures are not implemented in the Rust engine");
        }
        let mut event_keys = BTreeSet::new();
        for event in &self.stimulus.external_counts {
            if event.tick >= self.run.steps
                || event.neuron as usize >= self.network.neuron_count
                || event.count == 0
            {
                bail!("fixture contains an invalid external event");
            }
            if !event_keys.insert((event.tick, event.neuron)) {
                bail!("fixture contains a duplicate external event");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::TickFixture;
    use crate::reference::CpuEngine;

    fn fixture() -> TickFixture {
        TickFixture::load("fixtures/tiny-parity-v1.json").unwrap()
    }

    #[test]
    fn rust_float64_engine_matches_brian_validated_fixture() {
        let fixture = fixture();
        let connectome = fixture.connectome().unwrap();
        let schedule = fixture.event_schedule().unwrap();
        let mut engine = CpuEngine::new(
            &connectome,
            fixture.parameters,
            Some(&fixture.initial_state.v_mv),
            Some(&fixture.initial_state.g_mv),
            &fixture.overrides.zero_refractory,
            &fixture.overrides.silenced_sources,
        )
        .unwrap();
        let trace = engine.run_schedule(&schedule, true).unwrap();

        let mut spike_events = Vec::new();
        for (tick, state) in trace.iter().enumerate() {
            for (neuron, fired) in state.spikes.iter().enumerate() {
                if *fired != 0 {
                    spike_events.push((tick, neuron));
                }
            }
            for neuron in 0..connectome.neuron_count() {
                assert!(
                    (state.voltage_mv[neuron] - fixture.expected.v_end_mv[tick][neuron]).abs()
                        <= fixture.acceptance.state_abs_tol_mv
                );
                assert!(
                    (state.conductance_mv[neuron] - fixture.expected.g_end_mv[tick][neuron]).abs()
                        <= fixture.acceptance.state_abs_tol_mv
                );
            }
        }
        let expected: Vec<_> = fixture
            .expected
            .spike_events
            .iter()
            .map(|event| (event.tick, event.neuron))
            .collect();
        assert_eq!(spike_events, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rust_metal_engine_matches_fixture_spikes_and_float32_state() {
        use crate::metal_engine::MetalEngine;

        let fixture = fixture();
        let connectome = fixture.connectome().unwrap();
        let schedule = fixture.event_schedule().unwrap();
        let mut engine = MetalEngine::new(
            &connectome,
            fixture.parameters,
            Some(&fixture.initial_state.v_mv),
            Some(&fixture.initial_state.g_mv),
            &fixture.overrides.zero_refractory,
            &fixture.overrides.silenced_sources,
        )
        .unwrap();
        let trace = engine.run_recorded(&schedule).unwrap();

        let mut spike_events = Vec::new();
        for (tick, state) in trace.iter().enumerate() {
            for (neuron, fired) in state.spikes.iter().enumerate() {
                if *fired != 0 {
                    spike_events.push((tick, neuron));
                }
            }
            for neuron in 0..connectome.neuron_count() {
                assert!(
                    (state.voltage_mv[neuron] as f64 - fixture.expected.v_end_mv[tick][neuron])
                        .abs()
                        <= 1e-5
                );
                assert!(
                    (state.conductance_mv[neuron] as f64 - fixture.expected.g_end_mv[tick][neuron])
                        .abs()
                        <= 1e-5
                );
            }
        }
        let expected: Vec<_> = fixture
            .expected
            .spike_events
            .iter()
            .map(|event| (event.tick, event.neuron))
            .collect();
        assert_eq!(spike_events, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rust_backends_match_external_timing_delay_and_signed_edges() {
        use crate::metal_engine::MetalEngine;
        use crate::pack::ConnectomePack;
        use crate::parameters::ModelParameters;
        use crate::stimulus::EventSchedule;

        for signed_count in [4_i16, -4_i16] {
            let connectome =
                ConnectomePack::from_arrays([10, 20], [0, 1, 1], [1], [signed_count]).unwrap();
            let parameters = ModelParameters {
                delay_ms: 0.2,
                ..ModelParameters::default()
            };
            let schedule = EventSchedule::new(vec![0], vec![1, 0, 0, 0, 0], 5, 2).unwrap();
            let mut cpu = CpuEngine::new(&connectome, parameters, None, None, &[0], &[]).unwrap();
            let cpu_trace = cpu.run_schedule(&schedule, true).unwrap();
            let mut metal =
                MetalEngine::new(&connectome, parameters, None, None, &[0], &[]).unwrap();
            let metal_trace = metal.run_recorded(&schedule).unwrap();

            assert_eq!(cpu_trace[0].spikes, [0, 0]);
            assert_eq!(cpu_trace[1].spikes, [1, 0]);
            assert_eq!(cpu_trace[2].conductance_mv[1], 0.0);
            assert_eq!(cpu_trace[3].conductance_mv[1], signed_count as f64 * 0.275);
            for (expected, actual) in cpu_trace.iter().zip(&metal_trace) {
                assert_eq!(expected.spikes, actual.spikes);
                for neuron in 0..2 {
                    assert!(
                        (expected.voltage_mv[neuron] - actual.voltage_mv[neuron] as f64).abs()
                            <= 1e-5
                    );
                    assert!(
                        (expected.conductance_mv[neuron] - actual.conductance_mv[neuron] as f64)
                            .abs()
                            <= 1e-5
                    );
                }
            }
        }
    }
}
