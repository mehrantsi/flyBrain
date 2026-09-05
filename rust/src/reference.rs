use anyhow::{Result, bail};

use crate::pack::ConnectomePack;
use crate::parameters::ModelParameters;
use crate::stimulus::EventSchedule;

#[derive(Clone, Debug)]
pub struct StepState {
    pub spikes: Vec<u8>,
    pub voltage_mv: Vec<f64>,
    pub conductance_mv: Vec<f64>,
}

pub struct CpuEngine<'a> {
    connectome: &'a ConnectomePack,
    parameters: ModelParameters,
    voltage_mv: Vec<f64>,
    conductance_mv: Vec<f64>,
    refractory_remaining: Vec<i32>,
    refractory_lengths: Vec<i32>,
    silenced_sources: Vec<u8>,
    ring: Vec<f64>,
    ring_slots: usize,
    step_index: usize,
    spike_counts: Vec<u32>,
}

impl<'a> CpuEngine<'a> {
    pub fn new(
        connectome: &'a ConnectomePack,
        parameters: ModelParameters,
        initial_voltage_mv: Option<&[f64]>,
        initial_conductance_mv: Option<&[f64]>,
        zero_refractory: &[u32],
        silenced_sources: &[u32],
    ) -> Result<Self> {
        let parameters = parameters.validate()?;
        let neuron_count = connectome.neuron_ids.len();
        let voltage_mv = state_vector(
            initial_voltage_mv,
            neuron_count,
            parameters.resting_mv,
            "initial voltage",
        )?;
        let conductance_mv = state_vector(
            initial_conductance_mv,
            neuron_count,
            0.0,
            "initial conductance",
        )?;
        let mut refractory_lengths = vec![parameters.refractory_steps(); neuron_count];
        for &neuron in zero_refractory {
            let index = checked_index(neuron, neuron_count, "zero-refractory neuron")?;
            refractory_lengths[index] = 0;
        }
        let mut silenced = vec![0_u8; neuron_count];
        for &neuron in silenced_sources {
            let index = checked_index(neuron, neuron_count, "silenced source")?;
            silenced[index] = 1;
        }
        let ring_slots = parameters.delay_steps() + 1;
        let ring_length = ring_slots
            .checked_mul(neuron_count)
            .ok_or_else(|| anyhow::anyhow!("delay ring dimensions overflow"))?;
        Ok(Self {
            connectome,
            parameters,
            voltage_mv,
            conductance_mv,
            refractory_remaining: vec![0; neuron_count],
            refractory_lengths,
            silenced_sources: silenced,
            ring: vec![0.0; ring_length],
            ring_slots,
            step_index: 0,
            spike_counts: vec![0; neuron_count],
        })
    }

    pub fn step(&mut self, external_events: &[(u32, u8)]) -> Result<StepState> {
        let neuron_count = self.voltage_mv.len();
        let membrane_decay = self.parameters.membrane_decay();
        let synapse_decay = self.parameters.synapse_decay();
        let coupling = self.parameters.coupling();
        let mut spikes = vec![0_u8; neuron_count];

        for (neuron, fired) in spikes.iter_mut().enumerate() {
            if self.refractory_remaining[neuron] != 0 {
                continue;
            }
            let old_g = self.conductance_mv[neuron];
            self.conductance_mv[neuron] = old_g * synapse_decay;
            self.voltage_mv[neuron] = self.parameters.resting_mv
                + (self.voltage_mv[neuron] - self.parameters.resting_mv) * membrane_decay
                + old_g * coupling;
            if self.voltage_mv[neuron] > self.parameters.threshold_mv {
                *fired = 1;
                self.spike_counts[neuron] += 1;
            }
        }

        let target_slot = (self.step_index + self.parameters.delay_steps()) % self.ring_slots;
        let target_offset = target_slot * neuron_count;
        for (source, fired) in spikes.iter().copied().enumerate() {
            if fired == 0 || self.silenced_sources[source] != 0 {
                continue;
            }
            let begin = self.connectome.row_ptr[source] as usize;
            let end = self.connectome.row_ptr[source + 1] as usize;
            for edge in begin..end {
                let destination = self.connectome.destinations[edge] as usize;
                self.ring[target_offset + destination] +=
                    self.connectome.signed_counts[edge] as f64 * self.parameters.synapse_weight_mv;
            }
        }

        let current_slot = self.step_index % self.ring_slots;
        let current_offset = current_slot * neuron_count;
        for neuron in 0..neuron_count {
            self.conductance_mv[neuron] += self.ring[current_offset + neuron];
            self.ring[current_offset + neuron] = 0.0;
        }
        for &(neuron, count) in external_events {
            let index = checked_index(neuron, neuron_count, "external event target")?;
            self.voltage_mv[index] += count as f64 * self.parameters.external_weight_mv;
        }

        for (neuron, fired) in spikes.iter().copied().enumerate() {
            if fired != 0 {
                self.voltage_mv[neuron] = self.parameters.reset_mv;
                self.conductance_mv[neuron] = 0.0;
                self.refractory_remaining[neuron] = (self.refractory_lengths[neuron] - 1).max(0);
            } else if self.refractory_remaining[neuron] > 0 {
                self.refractory_remaining[neuron] -= 1;
            }
        }

        self.step_index += 1;
        Ok(StepState {
            spikes,
            voltage_mv: self.voltage_mv.clone(),
            conductance_mv: self.conductance_mv.clone(),
        })
    }

    pub fn run_schedule(
        &mut self,
        schedule: &EventSchedule,
        record_state: bool,
    ) -> Result<Vec<StepState>> {
        let mut recorded = Vec::with_capacity(if record_state { schedule.steps() } else { 0 });
        for step in 0..schedule.steps() {
            let events: Vec<_> = schedule.events_at(step).collect();
            let state = self.step(&events)?;
            if record_state {
                recorded.push(state);
            }
        }
        Ok(recorded)
    }

    pub fn voltage_mv(&self) -> &[f64] {
        &self.voltage_mv
    }

    pub fn conductance_mv(&self) -> &[f64] {
        &self.conductance_mv
    }

    pub fn spike_counts(&self) -> &[u32] {
        &self.spike_counts
    }
}

fn checked_index(index: u32, size: usize, name: &str) -> Result<usize> {
    let index = index as usize;
    if index >= size {
        bail!("{name} {index} is outside [0, {size})");
    }
    Ok(index)
}

fn state_vector(values: Option<&[f64]>, size: usize, default: f64, name: &str) -> Result<Vec<f64>> {
    let values = values.map_or_else(|| vec![default; size], <[f64]>::to_vec);
    if values.len() != size {
        bail!("{name} must have one value per neuron");
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("{name} must contain finite values");
    }
    Ok(values)
}
