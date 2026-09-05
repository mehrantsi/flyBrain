use std::collections::BTreeSet;
use std::env;

use anyhow::{Context, Result, bail};
use flybrain_engine::cns_pathway::{CnsPathway, PathwayControl};
use flybrain_engine::metal_engine::{MetalEngine, MetalStep};
use flybrain_engine::pack::ConnectomePack;
use flybrain_engine::parameters::ModelParameters;
use flybrain_engine::reference::{CpuEngine, StepState};

struct Args {
    pack: String,
    pathway: String,
    steps: usize,
    rate_hz: f64,
    seed: u64,
    mode: String,
}

impl Args {
    fn parse() -> Result<Self> {
        let values: Vec<_> = env::args().skip(1).collect();
        if !(5..=6).contains(&values.len()) {
            bail!(
                "usage: cns_numerics PACK PATHWAY STEPS RATE_HZ SEED [weighted|integer-arrivals|weighted-fma|integer-arrivals-fma]"
            );
        }
        Ok(Self {
            pack: values[0].clone(),
            pathway: values[1].clone(),
            steps: values[2].parse().context("parsing steps")?,
            rate_hz: values[3].parse().context("parsing rate_hz")?,
            seed: values[4].parse().context("parsing seed")?,
            mode: values.get(5).cloned().unwrap_or_else(|| "weighted".into()),
        })
    }
}

#[derive(Clone)]
struct F32State {
    spikes: Vec<u8>,
    voltage_mv: Vec<f32>,
    conductance_mv: Vec<f32>,
}

struct CpuF32Engine<'a> {
    connectome: &'a ConnectomePack,
    parameters: ModelParameters,
    voltage_mv: Vec<f32>,
    conductance_mv: Vec<f32>,
    refractory_remaining: Vec<i32>,
    refractory_lengths: Vec<i32>,
    silenced_sources: Vec<u8>,
    ring: ArrivalRing,
    fma: bool,
    ring_slots: usize,
    step_index: usize,
    spike_counts: Vec<u32>,
}

enum ArrivalRing {
    Weighted(Vec<f32>),
    Integer(Vec<i32>),
}

impl<'a> CpuF32Engine<'a> {
    fn new(
        connectome: &'a ConnectomePack,
        parameters: ModelParameters,
        zero_refractory: &[u32],
        silenced_sources: &[u32],
        mode: &str,
    ) -> Result<Self> {
        let parameters = parameters.validate()?;
        let neuron_count = connectome.neuron_count();
        let mut refractory_lengths = vec![parameters.refractory_steps(); neuron_count];
        for &neuron in zero_refractory {
            refractory_lengths[index(neuron, neuron_count)?] = 0;
        }
        let mut silenced = vec![0_u8; neuron_count];
        for &neuron in silenced_sources {
            silenced[index(neuron, neuron_count)?] = 1;
        }
        let ring_slots = parameters.delay_steps() + 1;
        Ok(Self {
            connectome,
            parameters,
            voltage_mv: vec![parameters.resting_mv as f32; neuron_count],
            conductance_mv: vec![0.0; neuron_count],
            refractory_remaining: vec![0; neuron_count],
            refractory_lengths,
            silenced_sources: silenced,
            ring: if mode.contains("integer-arrivals") {
                ArrivalRing::Integer(vec![0; ring_slots * neuron_count])
            } else {
                ArrivalRing::Weighted(vec![0.0; ring_slots * neuron_count])
            },
            fma: mode.contains("fma"),
            ring_slots,
            step_index: 0,
            spike_counts: vec![0; neuron_count],
        })
    }

    fn step(&mut self, external_events: &[(u32, u8)]) -> F32State {
        let neuron_count = self.voltage_mv.len();
        let membrane_decay = self.parameters.membrane_decay() as f32;
        let synapse_decay = self.parameters.synapse_decay() as f32;
        let coupling = self.parameters.coupling() as f32;
        let resting_mv = self.parameters.resting_mv as f32;
        let threshold_mv = self.parameters.threshold_mv as f32;
        let synapse_weight_mv = self.parameters.synapse_weight_mv as f32;
        let external_weight_mv = self.parameters.external_weight_mv as f32;
        let reset_mv = self.parameters.reset_mv as f32;
        let mut spikes = vec![0_u8; neuron_count];

        for (neuron, fired) in spikes.iter_mut().enumerate() {
            if self.refractory_remaining[neuron] != 0 {
                continue;
            }
            let old_g = self.conductance_mv[neuron];
            self.conductance_mv[neuron] = old_g * synapse_decay;
            let voltage_delta = self.voltage_mv[neuron] - resting_mv;
            self.voltage_mv[neuron] = if self.fma {
                old_g.mul_add(coupling, voltage_delta.mul_add(membrane_decay, resting_mv))
            } else {
                resting_mv + voltage_delta * membrane_decay + old_g * coupling
            };
            if self.voltage_mv[neuron] > threshold_mv {
                *fired = 1;
                self.spike_counts[neuron] += 1;
            }
        }

        let target_slot = (self.step_index + self.parameters.delay_steps()) % self.ring_slots;
        let target_offset = target_slot * neuron_count;
        match &mut self.ring {
            ArrivalRing::Weighted(ring) => {
                for (source, fired) in spikes.iter().copied().enumerate() {
                    if fired == 0 || self.silenced_sources[source] != 0 {
                        continue;
                    }
                    let begin = self.connectome.row_ptr[source] as usize;
                    let end = self.connectome.row_ptr[source + 1] as usize;
                    for edge in begin..end {
                        let destination = self.connectome.destinations[edge] as usize;
                        ring[target_offset + destination] +=
                            self.connectome.signed_counts[edge] as f32 * synapse_weight_mv;
                    }
                }
            }
            ArrivalRing::Integer(ring) => {
                for (source, fired) in spikes.iter().copied().enumerate() {
                    if fired == 0 || self.silenced_sources[source] != 0 {
                        continue;
                    }
                    let begin = self.connectome.row_ptr[source] as usize;
                    let end = self.connectome.row_ptr[source + 1] as usize;
                    for edge in begin..end {
                        let destination = self.connectome.destinations[edge] as usize;
                        ring[target_offset + destination] +=
                            self.connectome.signed_counts[edge] as i32;
                    }
                }
            }
        }

        let current_slot = self.step_index % self.ring_slots;
        let current_offset = current_slot * neuron_count;
        match &mut self.ring {
            ArrivalRing::Weighted(ring) => {
                for neuron in 0..neuron_count {
                    self.conductance_mv[neuron] += ring[current_offset + neuron];
                    ring[current_offset + neuron] = 0.0;
                }
            }
            ArrivalRing::Integer(ring) => {
                for neuron in 0..neuron_count {
                    self.conductance_mv[neuron] +=
                        ring[current_offset + neuron] as f32 * synapse_weight_mv;
                    ring[current_offset + neuron] = 0;
                }
            }
        }
        for &(neuron, count) in external_events {
            self.voltage_mv[neuron as usize] += count as f32 * external_weight_mv;
        }

        for (neuron, fired) in spikes.iter().copied().enumerate() {
            if fired != 0 {
                self.voltage_mv[neuron] = reset_mv;
                self.conductance_mv[neuron] = 0.0;
                self.refractory_remaining[neuron] = (self.refractory_lengths[neuron] - 1).max(0);
            } else if self.refractory_remaining[neuron] > 0 {
                self.refractory_remaining[neuron] -= 1;
            }
        }
        self.step_index += 1;
        F32State {
            spikes,
            voltage_mv: self.voltage_mv.clone(),
            conductance_mv: self.conductance_mv.clone(),
        }
    }
}

fn index(value: u32, size: usize) -> Result<usize> {
    let value = value as usize;
    if value >= size {
        bail!("index {value} is outside [0, {size})");
    }
    Ok(value)
}

fn event_mismatch(cpu: &[StepState], metal: &[MetalStep]) -> (usize, usize, Option<usize>) {
    let mut mismatch_ticks = 0;
    let mut mismatch_events = 0;
    let mut first = None;
    for (tick, (cpu, metal)) in cpu.iter().zip(metal).enumerate() {
        let cpu: BTreeSet<_> = cpu
            .spikes
            .iter()
            .enumerate()
            .filter_map(|(index, &fired)| (fired != 0).then_some(index))
            .collect();
        let metal: BTreeSet<_> = metal
            .spikes
            .iter()
            .enumerate()
            .filter_map(|(index, &fired)| (fired != 0).then_some(index))
            .collect();
        if cpu != metal {
            mismatch_ticks += 1;
            mismatch_events += cpu.symmetric_difference(&metal).count();
            first.get_or_insert(tick);
        }
    }
    (mismatch_ticks, mismatch_events, first)
}

fn event_mismatch_f32(cpu: &[F32State], metal: &[MetalStep]) -> (usize, usize, Option<usize>) {
    let mut mismatch_ticks = 0;
    let mut mismatch_events = 0;
    let mut first = None;
    for (tick, (cpu, metal)) in cpu.iter().zip(metal).enumerate() {
        let cpu: BTreeSet<_> = cpu
            .spikes
            .iter()
            .enumerate()
            .filter_map(|(index, &fired)| (fired != 0).then_some(index))
            .collect();
        let metal: BTreeSet<_> = metal
            .spikes
            .iter()
            .enumerate()
            .filter_map(|(index, &fired)| (fired != 0).then_some(index))
            .collect();
        if cpu != metal {
            mismatch_ticks += 1;
            mismatch_events += cpu.symmetric_difference(&metal).count();
            first.get_or_insert(tick);
        }
    }
    (mismatch_ticks, mismatch_events, first)
}

fn event_mismatch_cpu_f32(
    cpu: &[StepState],
    f32_cpu: &[F32State],
) -> (usize, usize, Option<usize>) {
    let mut mismatch_ticks = 0;
    let mut mismatch_events = 0;
    let mut first = None;
    for (tick, (cpu, f32_cpu)) in cpu.iter().zip(f32_cpu).enumerate() {
        let cpu: BTreeSet<_> = cpu
            .spikes
            .iter()
            .enumerate()
            .filter_map(|(index, &fired)| (fired != 0).then_some(index))
            .collect();
        let f32_cpu: BTreeSet<_> = f32_cpu
            .spikes
            .iter()
            .enumerate()
            .filter_map(|(index, &fired)| (fired != 0).then_some(index))
            .collect();
        if cpu != f32_cpu {
            mismatch_ticks += 1;
            mismatch_events += cpu.symmetric_difference(&f32_cpu).count();
            first.get_or_insert(tick);
        }
    }
    (mismatch_ticks, mismatch_events, first)
}

fn print_first_event_difference(
    label: &str,
    tick: Option<usize>,
    cpu: &[StepState],
    f32_cpu: &[F32State],
    metal: &[MetalStep],
    neuron_ids: &[u64],
) {
    let Some(tick) = tick else {
        return;
    };
    let cpu_spikes: BTreeSet<_> = cpu[tick]
        .spikes
        .iter()
        .enumerate()
        .filter_map(|(index, &fired)| (fired != 0).then_some(index))
        .collect();
    let f32_spikes: BTreeSet<_> = f32_cpu[tick]
        .spikes
        .iter()
        .enumerate()
        .filter_map(|(index, &fired)| (fired != 0).then_some(index))
        .collect();
    let metal_spikes: BTreeSet<_> = metal[tick]
        .spikes
        .iter()
        .enumerate()
        .filter_map(|(index, &fired)| (fired != 0).then_some(index))
        .collect();
    let differing: BTreeSet<_> = f32_spikes
        .symmetric_difference(&metal_spikes)
        .copied()
        .collect();
    eprintln!(
        "{label}_tick={tick} cpu_spikes={} f32_spikes={} metal_spikes={} differing_neurons={:?}",
        cpu_spikes.len(),
        f32_spikes.len(),
        metal_spikes.len(),
        differing
    );
    for neuron in differing {
        eprintln!(
            "{label}_neuron={} id={} cpu_f64_v={:.9e} f32_v={:.9e} metal_v={:.9e} cpu_f64_g={:.9e} f32_g={:.9e} metal_g={:.9e} threshold={:.9e}",
            neuron,
            neuron_ids[neuron],
            cpu[tick].voltage_mv[neuron],
            f32_cpu[tick].voltage_mv[neuron],
            metal[tick].voltage_mv[neuron],
            cpu[tick].conductance_mv[neuron],
            f32_cpu[tick].conductance_mv[neuron],
            metal[tick].conductance_mv[neuron],
            -45.0_f64,
        );
        if tick > 0 {
            let previous = tick - 1;
            eprintln!(
                "{label}_previous_tick={} neuron={} cpu_f64_v={:.9e} f32_v={:.9e} metal_v={:.9e} cpu_f64_g={:.9e} f32_g={:.9e} metal_g={:.9e}",
                previous,
                neuron,
                cpu[previous].voltage_mv[neuron],
                f32_cpu[previous].voltage_mv[neuron],
                metal[previous].voltage_mv[neuron],
                cpu[previous].conductance_mv[neuron],
                f32_cpu[previous].conductance_mv[neuron],
                metal[previous].conductance_mv[neuron],
            );
        }
    }
}

fn max_state_error(
    cpu: &[StepState],
    metal: &[MetalStep],
    f32_cpu: Option<&[F32State]>,
) -> (f64, usize, usize, f64, usize, usize, Option<usize>) {
    let mut max_v = 0.0;
    let mut max_v_tick = 0;
    let mut max_v_neuron = 0;
    let mut max_g = 0.0;
    let mut max_g_tick = 0;
    let mut max_g_neuron = 0;
    let mut first_large = None;
    for (tick, (cpu, metal)) in cpu.iter().zip(metal).enumerate() {
        let mut tick_max: f64 = 0.0;
        for neuron in 0..cpu.voltage_mv.len() {
            let v = (cpu.voltage_mv[neuron] - metal.voltage_mv[neuron] as f64).abs();
            if v > max_v {
                max_v = v;
                max_v_tick = tick;
                max_v_neuron = neuron;
            }
            let g = (cpu.conductance_mv[neuron] - metal.conductance_mv[neuron] as f64).abs();
            if g > max_g {
                max_g = g;
                max_g_tick = tick;
                max_g_neuron = neuron;
            }
            tick_max = tick_max.max(v).max(g);
        }
        if tick_max > 0.001 {
            first_large.get_or_insert(tick);
        }
    }
    if let Some(f32_cpu) = f32_cpu {
        let mut f32_v: f64 = 0.0;
        let mut f32_g: f64 = 0.0;
        for (state, metal) in f32_cpu.iter().zip(metal) {
            for neuron in 0..state.voltage_mv.len() {
                f32_v =
                    f32_v.max((state.voltage_mv[neuron] - metal.voltage_mv[neuron]).abs() as f64);
                f32_g = f32_g.max(
                    (state.conductance_mv[neuron] - metal.conductance_mv[neuron]).abs() as f64,
                );
            }
        }
        let final_cpu = f32_cpu.last().expect("f32 trace is nonempty");
        let final_metal = metal.last().expect("Metal trace is nonempty");
        let final_v = final_cpu
            .voltage_mv
            .iter()
            .zip(&final_metal.voltage_mv)
            .map(|(cpu, metal)| (*cpu - *metal).abs() as f64)
            .fold(0.0, f64::max);
        let final_g = final_cpu
            .conductance_mv
            .iter()
            .zip(&final_metal.conductance_mv)
            .map(|(cpu, metal)| (*cpu - *metal).abs() as f64)
            .fold(0.0, f64::max);
        eprintln!(
            "cpu_f32_vs_metal_max_trace_v={f32_v:.9e} cpu_f32_vs_metal_max_trace_g={f32_g:.9e} cpu_f32_vs_metal_final_v={final_v:.9e} cpu_f32_vs_metal_final_g={final_g:.9e}"
        );
    }
    let final_cpu = cpu.last().expect("CPU trace is nonempty");
    let final_metal = metal.last().expect("Metal trace is nonempty");
    let final_v = final_cpu
        .voltage_mv
        .iter()
        .zip(&final_metal.voltage_mv)
        .map(|(cpu, metal)| (*cpu - f64::from(*metal)).abs())
        .fold(0.0, f64::max);
    let final_g = final_cpu
        .conductance_mv
        .iter()
        .zip(&final_metal.conductance_mv)
        .map(|(cpu, metal)| (*cpu - f64::from(*metal)).abs())
        .fold(0.0, f64::max);
    eprintln!("cpu_f64_vs_metal_final_v={final_v:.9e} cpu_f64_vs_metal_final_g={final_g:.9e}");
    if let Some(f32_cpu) = f32_cpu {
        let final_f32 = f32_cpu.last().expect("f32 trace is nonempty");
        let f32_v = final_cpu
            .voltage_mv
            .iter()
            .zip(&final_f32.voltage_mv)
            .map(|(cpu, f32_cpu)| (*cpu - f64::from(*f32_cpu)).abs())
            .fold(0.0, f64::max);
        let f32_g = final_cpu
            .conductance_mv
            .iter()
            .zip(&final_f32.conductance_mv)
            .map(|(cpu, f32_cpu)| (*cpu - f64::from(*f32_cpu)).abs())
            .fold(0.0, f64::max);
        eprintln!("cpu_f64_vs_cpu_f32_final_v={f32_v:.9e} cpu_f64_vs_cpu_f32_final_g={f32_g:.9e}");
    }
    (
        max_v,
        max_v_tick,
        max_v_neuron,
        max_g,
        max_g_tick,
        max_g_neuron,
        first_large,
    )
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let pack = ConnectomePack::open(&args.pack)?;
    let pathway = CnsPathway::load(&args.pathway)?;
    let resolved = pathway.resolve(&pack)?;
    let parameters = ModelParameters::default();
    let schedule = resolved.schedule(
        PathwayControl::Intact,
        args.steps,
        args.rate_hz,
        args.seed,
        parameters,
        pack.neuron_count(),
    )?;
    let mut cpu = CpuEngine::new(
        &pack,
        parameters,
        None,
        None,
        &resolved.stimulus_indices,
        &[],
    )?;
    let cpu_trace = cpu.run_schedule(&schedule, true)?;
    let mut f32_cpu = CpuF32Engine::new(
        &pack,
        parameters,
        &resolved.stimulus_indices,
        &[],
        &args.mode,
    )?;
    let f32_trace: Vec<_> = (0..schedule.steps())
        .map(|step| f32_cpu.step(&schedule.events_at(step).collect::<Vec<_>>()))
        .collect();
    let mut metal = MetalEngine::new(
        &pack,
        parameters,
        None,
        None,
        &resolved.stimulus_indices,
        &[],
    )?;
    let metal_trace = metal.run_recorded(&schedule)?;
    let (mismatch_ticks, mismatch_events, first_mismatch) =
        event_mismatch(&cpu_trace, &metal_trace);
    let (f32_mismatch_ticks, f32_mismatch_events, f32_first_mismatch) =
        event_mismatch_f32(&f32_trace, &metal_trace);
    let (f64_f32_mismatch_ticks, f64_f32_mismatch_events, f64_f32_first_mismatch) =
        event_mismatch_cpu_f32(&cpu_trace, &f32_trace);
    print_first_event_difference(
        "f64_vs_metal",
        first_mismatch,
        &cpu_trace,
        &f32_trace,
        &metal_trace,
        &pack.neuron_ids,
    );
    print_first_event_difference(
        "f32_vs_metal",
        f32_first_mismatch,
        &cpu_trace,
        &f32_trace,
        &metal_trace,
        &pack.neuron_ids,
    );
    let (max_v, max_v_tick, max_v_neuron, max_g, max_g_tick, max_g_neuron, first_large) =
        max_state_error(&cpu_trace, &metal_trace, Some(&f32_trace));
    let count_mismatch = cpu
        .spike_counts()
        .iter()
        .zip(&f32_cpu.spike_counts)
        .filter(|(cpu, f32_cpu)| cpu != f32_cpu)
        .count();
    println!(
        "steps={} mode={} mismatch_ticks={} mismatch_events={} first_spike_mismatch={:?} f32_mismatch_ticks={} f32_mismatch_events={} f32_first_spike_mismatch={:?} f64_f32_mismatch_ticks={} f64_f32_mismatch_events={} f64_f32_first_spike_mismatch={:?} first_state_error_over_0.001={:?} max_v={:.9e}@tick{} neuron{}(id={}) max_g={:.9e}@tick{} neuron{}(id={}) f32_count_mismatch_neurons={}",
        args.steps,
        args.mode,
        mismatch_ticks,
        mismatch_events,
        first_mismatch,
        f32_mismatch_ticks,
        f32_mismatch_events,
        f32_first_mismatch,
        f64_f32_mismatch_ticks,
        f64_f32_mismatch_events,
        f64_f32_first_mismatch,
        first_large,
        max_v,
        max_v_tick,
        max_v_neuron,
        pack.neuron_ids[max_v_neuron],
        max_g,
        max_g_tick,
        max_g_neuron,
        pack.neuron_ids[max_g_neuron],
        count_mismatch,
    );
    Ok(())
}
