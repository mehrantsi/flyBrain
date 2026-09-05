use std::ffi::c_void;
use std::mem::{size_of, size_of_val};
use std::ptr::NonNull;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use crate::pack::ConnectomePack;
use crate::parameters::ModelParameters;
use crate::stimulus::EventSchedule;

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

const SHADER_SOURCE: &str = include_str!("../shaders/flybrain.metal");

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

enum ExternalInput<'a> {
    Dense {
        targets: &'a ProtocolObject<dyn MTLBuffer>,
        counts: &'a ProtocolObject<dyn MTLBuffer>,
        offset: u32,
        count: u32,
    },
    Sparse {
        targets_by_lane: &'a ProtocolObject<dyn MTLBuffer>,
        lanes: &'a ProtocolObject<dyn MTLBuffer>,
        counts: &'a ProtocolObject<dyn MTLBuffer>,
        offset: u32,
        count: u32,
    },
}

#[derive(Clone, Copy)]
struct DenseEvents<'a> {
    targets: &'a ProtocolObject<dyn MTLBuffer>,
    counts: &'a ProtocolObject<dyn MTLBuffer>,
    target_count: u32,
}

#[derive(Clone, Debug)]
pub struct MetalStep {
    pub spikes: Vec<u8>,
    pub voltage_mv: Vec<f32>,
    pub conductance_mv: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct MetalRun {
    pub elapsed: Duration,
    pub spike_counts: Vec<u32>,
    pub voltage_mv: Vec<f32>,
    pub conductance_mv: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalWindow {
    pub elapsed: Duration,
    pub spike_count_deltas: Vec<u32>,
}

pub type MetalWindowResult = MetalWindow;

impl MetalWindow {
    pub fn probe_spike_count_deltas(&self) -> &[u32] {
        &self.spike_count_deltas
    }
}

pub struct MetalEngine {
    device_name: String,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    decay_threshold: Pipeline,
    propagate_csr: Pipeline,
    decay_threshold_propagate_delayed: Pipeline,
    reset_decay_threshold: Pipeline,
    apply_external: Pipeline,
    apply_external_sparse: Pipeline,
    reset_store: Pipeline,
    decay_threadgroup_width: usize,
    propagation_threadgroup_width: usize,
    fused_threadgroup_width: usize,
    reset_decay_threadgroup_width: usize,
    reset_threadgroup_width: usize,
    row_ptr: Buffer,
    destinations: Buffer,
    signed_counts: Buffer,
    silenced_sources: Buffer,
    voltage: Buffer,
    conductance: Buffer,
    refractory_remaining: Buffer,
    refractory_lengths: Buffer,
    spikes: Buffer,
    spike_counts: Buffer,
    spike_ring: Buffer,
    arrivals: Buffer,
    external_targets: Buffer,
    sparse_event_lanes: Buffer,
    sparse_event_counts: Buffer,
    parameters: ModelParameters,
    neuron_count: u32,
    external_target_count: u32,
    sparse_event_capacity: usize,
    ring_size: u32,
    step_index: usize,
    allocated_bytes: usize,
}

impl MetalEngine {
    pub fn new(
        connectome: &ConnectomePack,
        parameters: ModelParameters,
        initial_voltage_mv: Option<&[f64]>,
        initial_conductance_mv: Option<&[f64]>,
        zero_refractory: &[u32],
        silenced_sources: &[u32],
    ) -> Result<Self> {
        let parameters = parameters.validate()?;
        autoreleasepool(|_| {
            let device = MTLCreateSystemDefaultDevice().context("no Metal device is available")?;
            let source = NSString::from_str(SHADER_SOURCE);
            let library = device.newLibraryWithSource_options_error(&source, None)?;
            let decay_threshold = pipeline(&device, &library, "decay_threshold")?;
            let propagate_csr = pipeline(&device, &library, "propagate_csr")?;
            let decay_threshold_propagate_delayed =
                pipeline(&device, &library, "decay_threshold_propagate_delayed")?;
            let reset_decay_threshold = pipeline(&device, &library, "reset_decay_threshold")?;
            let apply_external = pipeline(&device, &library, "apply_external")?;
            let apply_external_sparse = pipeline(&device, &library, "apply_external_sparse")?;
            let reset_store = pipeline(&device, &library, "reset_store")?;
            let queue = device
                .newCommandQueue()
                .context("Metal could not create a command queue")?;

            let neuron_count_usize = connectome.neuron_ids.len();
            let neuron_count = u32::try_from(neuron_count_usize)
                .context("neuron count does not fit in a Metal uint")?;
            let decay_threadgroup_width =
                preferred_threadgroup_width(&decay_threshold, neuron_count_usize);
            let propagation_threadgroup_width =
                preferred_threadgroup_width(&propagate_csr, neuron_count_usize);
            let fused_threadgroup_width =
                preferred_threadgroup_width(&decay_threshold_propagate_delayed, neuron_count_usize);
            let reset_decay_threadgroup_width =
                preferred_threadgroup_width(&reset_decay_threshold, neuron_count_usize);
            let reset_threadgroup_width =
                preferred_threadgroup_width(&reset_store, neuron_count_usize);
            let initial_voltage = f32_state(
                initial_voltage_mv,
                neuron_count_usize,
                parameters.resting_mv,
                "initial voltage",
            )?;
            let initial_conductance = f32_state(
                initial_conductance_mv,
                neuron_count_usize,
                0.0,
                "initial conductance",
            )?;
            let mut refractory_lengths = vec![parameters.refractory_steps(); neuron_count_usize];
            for &neuron in zero_refractory {
                refractory_lengths
                    [checked_index(neuron, neuron_count_usize, "zero-refractory neuron")?] = 0;
            }
            let mut silenced = vec![0_u8; neuron_count_usize];
            for &source in silenced_sources {
                silenced[checked_index(source, neuron_count_usize, "silenced source")?] = 1;
            }
            let ring_size = u32::try_from(parameters.delay_steps().max(1))?;
            let ring_length = neuron_count_usize
                .checked_mul(ring_size as usize)
                .context("delay ring dimensions overflow")?;

            let row_ptr = buffer_from_slice(&device, &connectome.row_ptr)?;
            let destinations = buffer_from_slice(&device, &connectome.destinations)?;
            let signed_counts = buffer_from_slice(&device, &connectome.signed_counts)?;
            let silenced_sources = buffer_from_slice(&device, &silenced)?;
            let voltage = buffer_from_slice(&device, &initial_voltage)?;
            let conductance = buffer_from_slice(&device, &initial_conductance)?;
            let refractory_remaining =
                buffer_from_slice(&device, &vec![0_i32; neuron_count_usize])?;
            let refractory_lengths_buffer = buffer_from_slice(&device, &refractory_lengths)?;
            let spikes = buffer_from_slice(&device, &vec![0_u8; neuron_count_usize])?;
            let spike_counts = buffer_from_slice(&device, &vec![0_u32; neuron_count_usize])?;
            let spike_ring = buffer_from_slice(&device, &vec![0_u8; ring_length])?;
            let arrivals = buffer_from_slice(&device, &vec![0_i32; neuron_count_usize])?;
            let external_targets = buffer_from_slice(&device, zero_refractory)?;
            let sparse_event_lanes = buffer_with_capacity::<u32>(&device, 1)?;
            let sparse_event_counts = buffer_with_capacity::<u8>(&device, 1)?;
            let external_target_count = u32::try_from(zero_refractory.len())?;

            let allocated_bytes = size_of_val(connectome.row_ptr.as_slice())
                + size_of_val(connectome.destinations.as_slice())
                + size_of_val(connectome.signed_counts.as_slice())
                + size_of_val(silenced.as_slice())
                + size_of_val(initial_voltage.as_slice())
                + size_of_val(initial_conductance.as_slice())
                + neuron_count_usize * size_of::<i32>()
                + size_of_val(refractory_lengths.as_slice())
                + neuron_count_usize
                + neuron_count_usize * size_of::<u32>()
                + ring_length
                + neuron_count_usize * size_of::<i32>();

            Ok(Self {
                device_name: device.name().to_string(),
                queue,
                decay_threshold,
                propagate_csr,
                decay_threshold_propagate_delayed,
                reset_decay_threshold,
                apply_external,
                apply_external_sparse,
                reset_store,
                decay_threadgroup_width,
                propagation_threadgroup_width,
                fused_threadgroup_width,
                reset_decay_threadgroup_width,
                reset_threadgroup_width,
                row_ptr,
                destinations,
                signed_counts,
                silenced_sources,
                voltage,
                conductance,
                refractory_remaining,
                refractory_lengths: refractory_lengths_buffer,
                spikes,
                spike_counts,
                spike_ring,
                arrivals,
                external_targets,
                sparse_event_lanes,
                sparse_event_counts,
                parameters,
                neuron_count,
                external_target_count,
                sparse_event_capacity: 1,
                ring_size,
                step_index: 0,
                allocated_bytes,
            })
        })
    }

    pub fn run_schedule(
        &mut self,
        schedule: &EventSchedule,
        chunk_steps: usize,
    ) -> Result<MetalRun> {
        let (targets, counts) = self.schedule_buffers(schedule)?;
        let target_count = u32::try_from(schedule.targets().len())?;
        let events = DenseEvents {
            targets: &targets,
            counts: &counts,
            target_count,
        };
        let chunk_steps = chunk_steps.max(1);
        let started = Instant::now();
        for chunk_start in (0..schedule.steps()).step_by(chunk_steps) {
            let chunk_end = (chunk_start + chunk_steps).min(schedule.steps());
            autoreleasepool(|_| -> Result<()> {
                let command_buffer = self
                    .queue
                    .commandBuffer()
                    .context("Metal could not create a command buffer")?;
                for event_step in chunk_start..chunk_end {
                    self.encode_tick_dense(
                        &command_buffer,
                        events,
                        event_step,
                        event_step != chunk_start,
                        event_step + 1 == chunk_end,
                    )?;
                    self.step_index += 1;
                }
                command_buffer.commit();
                command_buffer.waitUntilCompleted();
                if let Some(error) = command_buffer.error() {
                    bail!("Metal command buffer failed: {error}");
                }
                Ok(())
            })?;
        }
        let elapsed = started.elapsed();
        Ok(MetalRun {
            elapsed,
            spike_counts: read_buffer(&self.spike_counts, self.neuron_count as usize),
            voltage_mv: read_buffer(&self.voltage, self.neuron_count as usize),
            conductance_mv: read_buffer(&self.conductance, self.neuron_count as usize),
        })
    }

    pub fn run_recorded(&mut self, schedule: &EventSchedule) -> Result<Vec<MetalStep>> {
        let (targets, counts) = self.schedule_buffers(schedule)?;
        let target_count = u32::try_from(schedule.targets().len())?;
        let events = DenseEvents {
            targets: &targets,
            counts: &counts,
            target_count,
        };
        let mut trace = Vec::with_capacity(schedule.steps());
        for event_step in 0..schedule.steps() {
            autoreleasepool(|_| -> Result<()> {
                let command_buffer = self
                    .queue
                    .commandBuffer()
                    .context("Metal could not create a command buffer")?;
                self.encode_tick_dense(&command_buffer, events, event_step, false, true)?;
                command_buffer.commit();
                command_buffer.waitUntilCompleted();
                if let Some(error) = command_buffer.error() {
                    bail!("Metal command buffer failed: {error}");
                }
                Ok(())
            })?;
            self.step_index += 1;
            trace.push(MetalStep {
                spikes: read_buffer(&self.spikes, self.neuron_count as usize),
                voltage_mv: read_buffer(&self.voltage, self.neuron_count as usize),
                conductance_mv: read_buffer(&self.conductance, self.neuron_count as usize),
            });
        }
        Ok(trace)
    }

    pub fn run_window(
        &mut self,
        schedule: &EventSchedule,
        probe_neurons: &[u32],
    ) -> Result<MetalWindow> {
        validate_probe_neurons(probe_neurons, self.neuron_count as usize)?;
        let before = read_u32_indices(&self.spike_counts, probe_neurons);
        let (targets, counts) = self.schedule_buffers(schedule)?;
        let target_count = u32::try_from(schedule.targets().len())?;
        let events = DenseEvents {
            targets: &targets,
            counts: &counts,
            target_count,
        };
        let started = Instant::now();

        if schedule.steps() != 0 {
            autoreleasepool(|_| -> Result<()> {
                let command_buffer = self
                    .queue
                    .commandBuffer()
                    .context("Metal could not create a command buffer")?;
                for event_step in 0..schedule.steps() {
                    self.encode_tick_dense(
                        &command_buffer,
                        events,
                        event_step,
                        event_step != 0,
                        event_step + 1 == schedule.steps(),
                    )?;
                    self.step_index += 1;
                }
                command_buffer.commit();
                command_buffer.waitUntilCompleted();
                if let Some(error) = command_buffer.error() {
                    bail!("Metal command buffer failed: {error}");
                }
                Ok(())
            })?;
        }

        let after = read_u32_indices(&self.spike_counts, probe_neurons);
        let spike_count_deltas = before
            .into_iter()
            .zip(after)
            .map(|(before, after)| {
                after
                    .checked_sub(before)
                    .context("Metal probe spike count moved backwards")
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(MetalWindow {
            elapsed: started.elapsed(),
            spike_count_deltas,
        })
    }

    pub fn run_window_sparse(
        &mut self,
        steps: usize,
        step_offsets: &[u32],
        lanes: &[u32],
        counts: &[u8],
        probe_neurons: &[u32],
    ) -> Result<MetalWindow> {
        self.validate_sparse_window(steps, step_offsets, lanes, counts)?;
        validate_probe_neurons(probe_neurons, self.neuron_count as usize)?;
        let before = read_u32_indices(&self.spike_counts, probe_neurons);
        self.upload_sparse_events(lanes, counts)?;
        let started = Instant::now();

        if steps != 0 {
            autoreleasepool(|_| -> Result<()> {
                let command_buffer = self
                    .queue
                    .commandBuffer()
                    .context("Metal could not create a command buffer")?;
                for event_step in 0..steps {
                    let event_offset = step_offsets[event_step];
                    let event_count = step_offsets[event_step + 1] - event_offset;
                    self.encode_tick(
                        &command_buffer,
                        ExternalInput::Sparse {
                            targets_by_lane: &self.external_targets,
                            lanes: &self.sparse_event_lanes,
                            counts: &self.sparse_event_counts,
                            offset: event_offset,
                            count: event_count,
                        },
                        event_step != 0,
                        event_step + 1 == steps,
                    )?;
                    self.step_index += 1;
                }
                command_buffer.commit();
                command_buffer.waitUntilCompleted();
                if let Some(error) = command_buffer.error() {
                    bail!("Metal command buffer failed: {error}");
                }
                Ok(())
            })?;
        }

        let after = read_u32_indices(&self.spike_counts, probe_neurons);
        let spike_count_deltas = before
            .into_iter()
            .zip(after)
            .map(|(before, after)| {
                after
                    .checked_sub(before)
                    .context("Metal probe spike count moved backwards")
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(MetalWindow {
            elapsed: started.elapsed(),
            spike_count_deltas,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    pub fn total_spike_count(&self) -> u64 {
        let counts = unsafe {
            std::slice::from_raw_parts(
                self.spike_counts.contents().as_ptr().cast::<u32>(),
                self.neuron_count as usize,
            )
        };
        counts.iter().map(|&count| u64::from(count)).sum()
    }

    pub fn spiking_neuron_count(&self) -> usize {
        let counts = unsafe {
            std::slice::from_raw_parts(
                self.spike_counts.contents().as_ptr().cast::<u32>(),
                self.neuron_count as usize,
            )
        };
        counts.iter().filter(|&&count| count != 0).count()
    }

    pub fn mean_voltage_deviation_mv(&self, resting_mv: f64) -> f64 {
        let voltage = unsafe {
            std::slice::from_raw_parts(
                self.voltage.contents().as_ptr().cast::<f32>(),
                self.neuron_count as usize,
            )
        };
        voltage
            .iter()
            .map(|&value| f64::from(value) - resting_mv)
            .sum::<f64>()
            / f64::from(self.neuron_count)
    }

    fn schedule_buffers(&self, schedule: &EventSchedule) -> Result<(Buffer, Buffer)> {
        let device = self.queue.device();
        Ok((
            buffer_from_slice(&device, schedule.targets())?,
            buffer_from_slice(&device, schedule.counts())?,
        ))
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
                bail!("sparse stimulus step offsets are invalid")
            }
            let step_lanes = &lanes[offsets[0] as usize..offsets[1] as usize];
            if step_lanes
                .iter()
                .any(|&lane| lane >= self.external_target_count)
                || step_lanes.windows(2).any(|pair| pair[0] >= pair[1])
            {
                bail!("sparse stimulus lanes must be in range and strictly increasing per step")
            }
        }
        Ok(())
    }

    fn upload_sparse_events(&mut self, lanes: &[u32], counts: &[u8]) -> Result<()> {
        if lanes.len() > self.sparse_event_capacity {
            let capacity = lanes.len().next_power_of_two();
            let device = self.queue.device();
            self.sparse_event_lanes = buffer_with_capacity::<u32>(&device, capacity)?;
            self.sparse_event_counts = buffer_with_capacity::<u8>(&device, capacity)?;
            self.sparse_event_capacity = capacity;
        }
        copy_to_buffer(&self.sparse_event_lanes, lanes);
        copy_to_buffer(&self.sparse_event_counts, counts);
        Ok(())
    }

    fn encode_tick_dense(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        events: DenseEvents<'_>,
        event_step: usize,
        finalize_previous: bool,
        finalize_current: bool,
    ) -> Result<()> {
        let event_offset = u32::try_from(
            event_step
                .checked_mul(events.target_count as usize)
                .context("stimulus offset overflow")?,
        )?;
        self.encode_tick(
            command_buffer,
            ExternalInput::Dense {
                targets: events.targets,
                counts: events.counts,
                offset: event_offset,
                count: events.target_count,
            },
            finalize_previous,
            finalize_current,
        )
    }

    fn encode_tick(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        external_input: ExternalInput<'_>,
        finalize_previous: bool,
        finalize_current: bool,
    ) -> Result<()> {
        let neuron_count = self.neuron_count;
        let membrane_decay = self.parameters.membrane_decay() as f32;
        let synapse_decay = self.parameters.synapse_decay() as f32;
        let coupling = self.parameters.coupling() as f32;
        let resting_mv = self.parameters.resting_mv as f32;
        let threshold_mv = self.parameters.threshold_mv as f32;
        let synapse_weight_mv = self.parameters.synapse_weight_mv as f32;
        let external_weight_mv = self.parameters.external_weight_mv as f32;
        let reset_mv = self.parameters.reset_mv as f32;

        let ring_slot = (self.step_index % self.ring_size as usize) as u32;
        let (delayed_buffer, delayed_offset) = if self.parameters.delay_steps() == 0 {
            (&*self.spikes, 0)
        } else {
            (
                &*self.spike_ring,
                ring_slot as usize * neuron_count as usize * size_of::<u8>(),
            )
        };
        let propagation_pending = if finalize_previous {
            let previous_ring_slot =
                ((self.step_index + self.ring_size as usize - 1) % self.ring_size as usize) as u32;
            let encoder = command_buffer
                .computeCommandEncoder()
                .context("Metal could not create the reset-decay encoder")?;
            encoder.setComputePipelineState(&self.reset_decay_threshold);
            unsafe {
                bind(&encoder, 0, &self.voltage, 0);
                bind(&encoder, 1, &self.conductance, 0);
                bind(&encoder, 2, &self.refractory_remaining, 0);
                bind(&encoder, 3, &self.refractory_lengths, 0);
                bind(&encoder, 4, &self.spikes, 0);
                bind(&encoder, 5, &self.spike_counts, 0);
                bind(&encoder, 6, &self.spike_ring, 0);
                bind(&encoder, 7, &self.arrivals, 0);
                scalar(&encoder, 8, &neuron_count);
                scalar(&encoder, 9, &previous_ring_slot);
                scalar(&encoder, 10, &reset_mv);
                scalar(&encoder, 11, &synapse_weight_mv);
                scalar(&encoder, 12, &resting_mv);
                scalar(&encoder, 13, &threshold_mv);
                scalar(&encoder, 14, &membrane_decay);
                scalar(&encoder, 15, &synapse_decay);
                scalar(&encoder, 16, &coupling);
            }
            dispatch_with_threadgroup(
                &encoder,
                neuron_count as usize,
                self.reset_decay_threadgroup_width,
            );
            encoder.endEncoding();
            true
        } else if self.parameters.delay_steps() == 0 {
            let encoder = command_buffer
                .computeCommandEncoder()
                .context("Metal could not create the decay encoder")?;
            encoder.setComputePipelineState(&self.decay_threshold);
            unsafe {
                bind(&encoder, 0, &self.voltage, 0);
                bind(&encoder, 1, &self.conductance, 0);
                bind(&encoder, 2, &self.refractory_remaining, 0);
                bind(&encoder, 3, &self.spikes, 0);
                bind(&encoder, 4, &self.spike_counts, 0);
                scalar(&encoder, 5, &neuron_count);
                scalar(&encoder, 6, &resting_mv);
                scalar(&encoder, 7, &threshold_mv);
                scalar(&encoder, 8, &membrane_decay);
                scalar(&encoder, 9, &synapse_decay);
                scalar(&encoder, 10, &coupling);
            }
            dispatch_with_threadgroup(
                &encoder,
                neuron_count as usize,
                self.decay_threadgroup_width,
            );
            encoder.endEncoding();
            true
        } else {
            let encoder = command_buffer
                .computeCommandEncoder()
                .context("Metal could not create the fused neural encoder")?;
            encoder.setComputePipelineState(&self.decay_threshold_propagate_delayed);
            unsafe {
                bind(&encoder, 0, &self.voltage, 0);
                bind(&encoder, 1, &self.conductance, 0);
                bind(&encoder, 2, &self.refractory_remaining, 0);
                bind(&encoder, 3, &self.spikes, 0);
                bind(&encoder, 4, &self.spike_counts, 0);
                bind(&encoder, 5, &self.row_ptr, 0);
                bind(&encoder, 6, &self.destinations, 0);
                bind(&encoder, 7, &self.signed_counts, 0);
                bind(&encoder, 8, delayed_buffer, delayed_offset);
                bind(&encoder, 9, &self.silenced_sources, 0);
                bind(&encoder, 10, &self.arrivals, 0);
                scalar(&encoder, 11, &neuron_count);
                scalar(&encoder, 12, &resting_mv);
                scalar(&encoder, 13, &threshold_mv);
                scalar(&encoder, 14, &membrane_decay);
                scalar(&encoder, 15, &synapse_decay);
                scalar(&encoder, 16, &coupling);
            }
            dispatch_with_threadgroup(
                &encoder,
                neuron_count as usize,
                self.fused_threadgroup_width,
            );
            encoder.endEncoding();
            false
        };

        if propagation_pending {
            let encoder = command_buffer
                .computeCommandEncoder()
                .context("Metal could not create the propagation encoder")?;
            encoder.setComputePipelineState(&self.propagate_csr);
            unsafe {
                bind(&encoder, 0, &self.row_ptr, 0);
                bind(&encoder, 1, &self.destinations, 0);
                bind(&encoder, 2, &self.signed_counts, 0);
                bind(&encoder, 3, delayed_buffer, delayed_offset);
                bind(&encoder, 4, &self.silenced_sources, 0);
                bind(&encoder, 5, &self.arrivals, 0);
                scalar(&encoder, 6, &neuron_count);
            }
            dispatch_with_threadgroup(
                &encoder,
                neuron_count as usize,
                self.propagation_threadgroup_width,
            );
            encoder.endEncoding();
        }

        match external_input {
            ExternalInput::Dense {
                targets,
                counts,
                offset,
                count,
            } if count != 0 => {
                let encoder = command_buffer
                    .computeCommandEncoder()
                    .context("Metal could not create the external-input encoder")?;
                encoder.setComputePipelineState(&self.apply_external);
                unsafe {
                    bind(&encoder, 0, &self.voltage, 0);
                    bind(&encoder, 1, targets, 0);
                    bind(&encoder, 2, counts, 0);
                    scalar(&encoder, 3, &offset);
                    scalar(&encoder, 4, &count);
                    scalar(&encoder, 5, &external_weight_mv);
                }
                dispatch(&encoder, &self.apply_external, count as usize);
                encoder.endEncoding();
            }
            ExternalInput::Sparse {
                targets_by_lane,
                lanes,
                counts,
                offset,
                count,
            } if count != 0 => {
                let encoder = command_buffer
                    .computeCommandEncoder()
                    .context("Metal could not create the sparse external-input encoder")?;
                encoder.setComputePipelineState(&self.apply_external_sparse);
                unsafe {
                    bind(&encoder, 0, &self.voltage, 0);
                    bind(&encoder, 1, targets_by_lane, 0);
                    bind(&encoder, 2, lanes, 0);
                    bind(&encoder, 3, counts, 0);
                    scalar(&encoder, 4, &offset);
                    scalar(&encoder, 5, &count);
                    scalar(&encoder, 6, &external_weight_mv);
                }
                dispatch(&encoder, &self.apply_external_sparse, count as usize);
                encoder.endEncoding();
            }
            ExternalInput::Dense { .. } | ExternalInput::Sparse { .. } => {}
        }

        if finalize_current {
            let encoder = command_buffer
                .computeCommandEncoder()
                .context("Metal could not create the reset encoder")?;
            encoder.setComputePipelineState(&self.reset_store);
            unsafe {
                bind(&encoder, 0, &self.voltage, 0);
                bind(&encoder, 1, &self.conductance, 0);
                bind(&encoder, 2, &self.refractory_remaining, 0);
                bind(&encoder, 3, &self.refractory_lengths, 0);
                bind(&encoder, 4, &self.spikes, 0);
                bind(&encoder, 5, &self.spike_ring, 0);
                bind(&encoder, 6, &self.arrivals, 0);
                scalar(&encoder, 7, &neuron_count);
                scalar(&encoder, 8, &ring_slot);
                scalar(&encoder, 9, &reset_mv);
                scalar(&encoder, 10, &synapse_weight_mv);
            }
            dispatch_with_threadgroup(
                &encoder,
                neuron_count as usize,
                self.reset_threadgroup_width,
            );
            encoder.endEncoding();
        }
        Ok(())
    }
}

fn pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Pipeline> {
    let name = NSString::from_str(name);
    let function = library
        .newFunctionWithName(&name)
        .with_context(|| format!("Metal function {name} is missing"))?;
    Ok(device.newComputePipelineStateWithFunction_error(&function)?)
}

fn buffer_from_slice<T>(device: &ProtocolObject<dyn MTLDevice>, values: &[T]) -> Result<Buffer> {
    let byte_length = size_of_val(values);
    let options = MTLResourceOptions::StorageModeShared;
    let buffer = if byte_length == 0 {
        device.newBufferWithLength_options(1, options)
    } else {
        let pointer = NonNull::new(values.as_ptr() as *mut c_void).unwrap();
        unsafe { device.newBufferWithBytes_length_options(pointer, byte_length, options) }
    };
    buffer.context("Metal could not allocate a shared buffer")
}

fn buffer_with_capacity<T>(
    device: &ProtocolObject<dyn MTLDevice>,
    capacity: usize,
) -> Result<Buffer> {
    let byte_length = capacity
        .checked_mul(size_of::<T>())
        .context("Metal buffer capacity overflow")?
        .max(1);
    device
        .newBufferWithLength_options(byte_length, MTLResourceOptions::StorageModeShared)
        .context("Metal could not allocate a shared buffer")
}

fn copy_to_buffer<T: Copy>(buffer: &ProtocolObject<dyn MTLBuffer>, values: &[T]) {
    if values.is_empty() {
        return;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            buffer.contents().as_ptr().cast::<T>(),
            values.len(),
        )
    }
}

unsafe fn bind(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: usize,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    offset: usize,
) {
    unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), offset, index) };
}

unsafe fn scalar<T>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: usize,
    value: &T,
) {
    let pointer = NonNull::from(value).cast::<c_void>();
    unsafe { encoder.setBytes_length_atIndex(pointer, size_of::<T>(), index) };
}

fn dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    width: usize,
) {
    if width == 0 {
        return;
    }
    dispatch_with_threadgroup(encoder, width, preferred_threadgroup_width(pipeline, width));
}

fn preferred_threadgroup_width(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    width: usize,
) -> usize {
    if width == 0 {
        return 1;
    }
    let execution_width = pipeline.threadExecutionWidth();
    let maximum_width = pipeline.maxTotalThreadsPerThreadgroup().min(width);
    if maximum_width < execution_width {
        maximum_width
    } else {
        (maximum_width.min(256) / execution_width) * execution_width
    }
}

fn dispatch_with_threadgroup(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    width: usize,
    threadgroup_width: usize,
) {
    if width == 0 {
        return;
    }
    let grid = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let threads = MTLSize {
        width: threadgroup_width,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid, threads);
}

fn read_buffer<T: Copy>(buffer: &ProtocolObject<dyn MTLBuffer>, length: usize) -> Vec<T> {
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<T>(), length).to_vec() }
}

fn validate_probe_neurons(neurons: &[u32], neuron_count: usize) -> Result<()> {
    for &neuron in neurons {
        checked_index(neuron, neuron_count, "probe neuron")?;
    }
    Ok(())
}

fn read_u32_indices(buffer: &ProtocolObject<dyn MTLBuffer>, indices: &[u32]) -> Vec<u32> {
    let values = buffer.contents().as_ptr().cast::<u32>();
    indices
        .iter()
        .map(|&index| unsafe { *values.add(index as usize) })
        .collect()
}

fn f32_state(values: Option<&[f64]>, size: usize, default: f64, name: &str) -> Result<Vec<f32>> {
    let values = values.map_or_else(|| vec![default; size], <[f64]>::to_vec);
    if values.len() != size {
        bail!("{name} must have one value per neuron");
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("{name} must contain finite values");
    }
    Ok(values.into_iter().map(|value| value as f32).collect())
}

fn checked_index(index: u32, size: usize, name: &str) -> Result<usize> {
    let index = index as usize;
    if index >= size {
        bail!("{name} {index} is outside [0, {size})");
    }
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::{MetalEngine, read_buffer};
    use crate::pack::ConnectomePack;
    use crate::parameters::ModelParameters;
    use crate::stimulus::EventSchedule;

    #[test]
    fn consecutive_windows_preserve_state_and_match_combined_schedule() {
        let connectome = ConnectomePack::from_arrays([0_u64], [0_u32, 0], [], []).unwrap();
        let parameters = ModelParameters::default();
        let first = EventSchedule::new(vec![0], vec![1], 1, 1).unwrap();
        let second = EventSchedule::new(vec![0], vec![0], 1, 1).unwrap();
        let combined = EventSchedule::new(vec![0], vec![1, 0], 2, 1).unwrap();

        let mut streamed =
            MetalEngine::new(&connectome, parameters, None, None, &[0], &[]).unwrap();
        let first_result = streamed.run_window(&first, &[0]).unwrap();
        let second_result = streamed.run_window(&second, &[0]).unwrap();

        assert_eq!(first_result.spike_count_deltas, [0]);
        assert_eq!(second_result.spike_count_deltas, [1]);
        assert_eq!(streamed.total_spike_count(), 1);
        assert_eq!(
            first_result
                .spike_count_deltas
                .iter()
                .zip(&second_result.spike_count_deltas)
                .map(|(first, second)| first + second)
                .collect::<Vec<_>>(),
            [1]
        );

        let mut combined_engine =
            MetalEngine::new(&connectome, parameters, None, None, &[0], &[]).unwrap();
        let combined_result = combined_engine.run_window(&combined, &[0]).unwrap();
        assert_eq!(combined_result.spike_count_deltas, [1]);
        assert_eq!(combined_engine.total_spike_count(), 1);
    }

    #[test]
    fn sparse_external_events_match_dense_schedule_exactly() {
        let connectome = ConnectomePack::from_arrays([10_u64, 20], [0_u32, 0, 0], [], []).unwrap();
        let parameters = ModelParameters::default();
        let dense =
            EventSchedule::new(vec![0, 1], vec![1, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0], 6, 2).unwrap();
        let mut dense_engine =
            MetalEngine::new(&connectome, parameters, None, None, &[0, 1], &[]).unwrap();
        let dense_result = dense_engine.run_window(&dense, &[0, 1]).unwrap();

        let mut sparse_engine =
            MetalEngine::new(&connectome, parameters, None, None, &[0, 1], &[]).unwrap();
        let sparse_result = sparse_engine
            .run_window_sparse(6, &[0, 1, 1, 2, 2, 3, 3], &[0, 1, 0], &[1, 2, 1], &[0, 1])
            .unwrap();

        assert_eq!(
            sparse_result.spike_count_deltas,
            dense_result.spike_count_deltas
        );
        assert_eq!(
            sparse_engine.total_spike_count(),
            dense_engine.total_spike_count()
        );
        assert_eq!(
            read_buffer::<f32>(&sparse_engine.voltage, 2),
            read_buffer::<f32>(&dense_engine.voltage, 2)
        );
        assert_eq!(
            read_buffer::<f32>(&sparse_engine.conductance, 2),
            read_buffer::<f32>(&dense_engine.conductance, 2)
        );
        assert!(
            sparse_engine
                .run_window_sparse(1, &[0, 2], &[0, 0], &[1, 1], &[])
                .is_err()
        );
    }
}
