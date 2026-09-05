use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::Serialize;

use crate::pack::ConnectomePack;
use crate::parameters::ModelParameters;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrowserWindow {
    pub elapsed: Duration,
    pub spike_count_deltas: Vec<u32>,
}

pub type BrowserWindowResult = BrowserWindow;

impl BrowserWindow {
    pub fn probe_spike_count_deltas(&self) -> &[u32] {
        &self.spike_count_deltas
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserEngineInit {
    neuron_count: u32,
    edge_count: u32,
    ring_size: u32,
    delay_steps: u32,
    row_ptr_ptr: usize,
    destinations_ptr: usize,
    signed_counts_ptr: usize,
    voltage_ptr: usize,
    conductance_ptr: usize,
    refractory_remaining_ptr: usize,
    refractory_lengths_ptr: usize,
    spikes_ptr: usize,
    spike_counts_ptr: usize,
    spike_ring_ptr: usize,
    arrivals_ptr: usize,
    silenced_sources_ptr: usize,
    external_targets_ptr: usize,
    external_target_count: u32,
    resting_mv: f32,
    reset_mv: f32,
    threshold_mv: f32,
    membrane_decay: f32,
    synapse_decay: f32,
    coupling: f32,
    synapse_weight_mv: f32,
    external_weight_mv: f32,
}

unsafe extern "C" {
    fn fly_gpu_create(json_ptr: usize, json_len: usize) -> i32;
    fn fly_gpu_window(
        handle: i32,
        steps: usize,
        offsets_ptr: usize,
        lanes_ptr: usize,
        counts_ptr: usize,
        event_len: usize,
        probes_ptr: usize,
        probe_len: usize,
        result_ptr: usize,
    ) -> i32;
    fn fly_gpu_destroy(handle: i32);
}

pub struct BrowserEngine {
    handle: i32,
    external_targets: Vec<u32>,
    neuron_count: u32,
    allocated_bytes: usize,
    total_spike_count: u64,
    spiking_neuron_count: usize,
    mean_voltage_mv: f64,
}

impl BrowserEngine {
    pub fn new(
        connectome: &ConnectomePack,
        parameters: ModelParameters,
        initial_voltage_mv: Option<&[f64]>,
        initial_conductance_mv: Option<&[f64]>,
        zero_refractory: &[u32],
        silenced_sources: &[u32],
    ) -> Result<Self> {
        let parameters = parameters.validate()?;
        let neuron_count = connectome.neuron_count();
        let neuron_count_u32 = u32::try_from(neuron_count)
            .context("neuron count does not fit in a browser GPU uint")?;
        let edge_count = u32::try_from(connectome.edge_count())
            .context("edge count does not fit in a browser GPU uint")?;
        let ring_size = u32::try_from(parameters.delay_steps().max(1))?;
        let voltage = f32_state(
            initial_voltage_mv,
            neuron_count,
            parameters.resting_mv,
            "initial voltage",
        )?;
        let conductance = f32_state(
            initial_conductance_mv,
            neuron_count,
            0.0,
            "initial conductance",
        )?;
        let mut refractory_lengths = vec![parameters.refractory_steps(); neuron_count];
        for &neuron in zero_refractory {
            refractory_lengths[checked_index(neuron, neuron_count, "zero-refractory neuron")?] = 0;
        }
        let mut silenced = vec![0_u32; neuron_count];
        for &neuron in silenced_sources {
            silenced[checked_index(neuron, neuron_count, "silenced source")?] = 1;
        }
        let ring_length = neuron_count
            .checked_mul(ring_size as usize)
            .context("spike ring dimensions overflow")?;
        let refractory_remaining = vec![0; neuron_count];
        let spikes = vec![0; neuron_count];
        let spike_counts = vec![0; neuron_count];
        let spike_ring = vec![0; ring_length];
        let arrivals = vec![0; neuron_count];
        let external_targets = zero_refractory.to_vec();
        let allocated_bytes = gpu_bytes(neuron_count, ring_length, connectome)?;
        let metadata = BrowserEngineInit {
            neuron_count: neuron_count_u32,
            edge_count,
            ring_size,
            delay_steps: u32::try_from(parameters.delay_steps())?,
            row_ptr_ptr: connectome.row_ptr.as_ptr() as usize,
            destinations_ptr: connectome.destinations.as_ptr() as usize,
            signed_counts_ptr: connectome.signed_counts.as_ptr() as usize,
            voltage_ptr: voltage.as_ptr() as usize,
            conductance_ptr: conductance.as_ptr() as usize,
            refractory_remaining_ptr: refractory_remaining.as_ptr() as usize,
            refractory_lengths_ptr: refractory_lengths.as_ptr() as usize,
            spikes_ptr: spikes.as_ptr() as usize,
            spike_counts_ptr: spike_counts.as_ptr() as usize,
            spike_ring_ptr: spike_ring.as_ptr() as usize,
            arrivals_ptr: arrivals.as_ptr() as usize,
            silenced_sources_ptr: silenced.as_ptr() as usize,
            external_targets_ptr: external_targets.as_ptr() as usize,
            external_target_count: u32::try_from(external_targets.len())?,
            resting_mv: parameters.resting_mv as f32,
            reset_mv: parameters.reset_mv as f32,
            threshold_mv: parameters.threshold_mv as f32,
            membrane_decay: parameters.membrane_decay() as f32,
            synapse_decay: parameters.synapse_decay() as f32,
            coupling: parameters.coupling() as f32,
            synapse_weight_mv: parameters.synapse_weight_mv as f32,
            external_weight_mv: parameters.external_weight_mv as f32,
        };
        let json = serde_json::to_vec(&metadata).context("encoding browser GPU metadata")?;
        let handle = unsafe { fly_gpu_create(json.as_ptr() as usize, json.len()) };
        if handle < 0 {
            bail!("browser WebGPU initialization failed")
        }
        Ok(Self {
            handle,
            external_targets,
            neuron_count: neuron_count_u32,
            allocated_bytes,
            total_spike_count: 0,
            spiking_neuron_count: 0,
            mean_voltage_mv: parameters.resting_mv,
        })
    }

    pub fn run_window_sparse(
        &mut self,
        steps: usize,
        step_offsets: &[u32],
        lanes: &[u32],
        counts: &[u8],
        probe_neurons: &[u32],
    ) -> Result<BrowserWindow> {
        self.validate_sparse_window(steps, step_offsets, lanes, counts)?;
        for &probe in probe_neurons {
            checked_index(probe, self.neuron_count as usize, "probe neuron")?;
        }
        let mut result = vec![0_u32; 5 + probe_neurons.len()];
        let started = Instant::now();
        let status = unsafe {
            fly_gpu_window(
                self.handle,
                steps,
                step_offsets.as_ptr() as usize,
                lanes.as_ptr() as usize,
                counts.as_ptr() as usize,
                lanes.len(),
                probe_neurons.as_ptr() as usize,
                probe_neurons.len(),
                result.as_mut_ptr() as usize,
            )
        };
        if status != 0 {
            bail!("browser WebGPU window failed with status {status}")
        }
        if result[4] as usize != probe_neurons.len() {
            bail!("browser WebGPU returned an invalid probe result length")
        }
        self.total_spike_count = u64::from(result[0]) | (u64::from(result[1]) << 32);
        self.spiking_neuron_count = result[2] as usize;
        self.mean_voltage_mv = f64::from(f32::from_bits(result[3]));
        Ok(BrowserWindow {
            elapsed: started.elapsed(),
            spike_count_deltas: result[5..].to_vec(),
        })
    }

    pub fn device_name(&self) -> &str {
        "WebGPU"
    }

    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    pub fn total_spike_count(&self) -> u64 {
        self.total_spike_count
    }

    pub fn spiking_neuron_count(&self) -> usize {
        self.spiking_neuron_count
    }

    pub fn mean_voltage_deviation_mv(&self, resting_mv: f64) -> f64 {
        self.mean_voltage_mv - resting_mv
    }

    fn validate_sparse_window(
        &self,
        steps: usize,
        step_offsets: &[u32],
        lanes: &[u32],
        counts: &[u8],
    ) -> Result<()> {
        if lanes.len() > u32::MAX as usize
            || step_offsets.len() != steps + 1
            || step_offsets.first().copied() != Some(0)
            || step_offsets.last().copied() != Some(lanes.len() as u32)
            || lanes.len() != counts.len()
            || counts.contains(&0)
        {
            bail!("sparse stimulus window dimensions or counts are invalid")
        }
        for offsets in step_offsets.windows(2) {
            if offsets[0] > offsets[1] || offsets[1] as usize > lanes.len() {
                bail!("sparse stimulus offsets are invalid")
            }
            let step_lanes = &lanes[offsets[0] as usize..offsets[1] as usize];
            if step_lanes
                .iter()
                .any(|&lane| lane as usize >= self.external_targets.len())
                || step_lanes.windows(2).any(|pair| pair[0] >= pair[1])
            {
                bail!("sparse stimulus lanes must be in range and strictly increasing per step")
            }
        }
        Ok(())
    }
}

impl Drop for BrowserEngine {
    fn drop(&mut self) {
        if self.handle >= 0 {
            unsafe { fly_gpu_destroy(self.handle) };
        }
    }
}

fn gpu_bytes(
    neuron_count: usize,
    ring_length: usize,
    connectome: &ConnectomePack,
) -> Result<usize> {
    let state = neuron_count
        .checked_mul(32)
        .context("browser state buffer dimensions overflow")?;
    let ring = ring_length
        .checked_mul(std::mem::size_of::<u32>())
        .context("browser ring buffer dimensions overflow")?;
    let row_ptr = connectome
        .row_ptr
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .context("browser row pointer dimensions overflow")?;
    let destinations = connectome
        .destinations
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .context("browser destination dimensions overflow")?;
    let weights = connectome
        .signed_counts
        .len()
        .checked_mul(std::mem::size_of::<i32>())
        .context("browser weight dimensions overflow")?;
    let arrivals = neuron_count
        .checked_mul(std::mem::size_of::<i32>())
        .context("browser arrival dimensions overflow")?;
    state
        .checked_add(ring)
        .and_then(|value| value.checked_add(row_ptr))
        .and_then(|value| value.checked_add(destinations))
        .and_then(|value| value.checked_add(weights))
        .and_then(|value| value.checked_add(arrivals))
        .context("browser GPU allocation estimate overflow")
}

fn checked_index(index: u32, size: usize, name: &str) -> Result<usize> {
    let index = index as usize;
    if index >= size {
        bail!("{name} {index} is outside 0..{size}")
    }
    Ok(index)
}

fn f32_state(values: Option<&[f64]>, size: usize, default: f64, name: &str) -> Result<Vec<f32>> {
    let values = values.map_or_else(|| vec![default; size], <[f64]>::to_vec);
    if values.len() != size {
        bail!("{name} must contain one value per neuron")
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("{name} must contain finite values")
    }
    Ok(values.into_iter().map(|value| value as f32).collect())
}
