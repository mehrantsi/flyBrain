struct State {
  voltage: f32,
  conductance: f32,
  refractory: i32,
  refractory_length: i32,
  spikes: u32,
  spike_count: u32,
  silenced: u32,
  padding: u32,
}

const SOURCE_LANES: u32 = 256u;
const MAX_SOURCE_GROUPS: u32 = 65535u;

struct TickParams {
  neuron_count: u32,
  ring_size: u32,
  ring_slot: u32,
  event_offset: u32,
  event_count: u32,
  delay_steps: u32,
  target_count: u32,
  active_slot: u32,
  resting_mv: f32,
  reset_mv: f32,
  threshold_mv: f32,
  membrane_decay: f32,
  synapse_decay: f32,
  coupling: f32,
  synapse_weight_mv: f32,
  external_weight_mv: f32,
}

@group(0) @binding(0) var<storage, read_write> states: array<State>;
@group(0) @binding(1) var<storage, read_write> spike_ring: array<u32>;
@group(0) @binding(2) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(3) var<storage, read> destinations: array<u32>;
@group(0) @binding(4) var<storage, read> signed_counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> arrivals: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read> events: array<u32>;
@group(0) @binding(7) var<uniform> tick: TickParams;
@group(0) @binding(8) var<storage, read_write> dispatch_args: array<atomic<u32>>;

fn queue_source(source: u32) {
  let other = 1u - tick.active_slot;
  if (source == 0u) {
    atomicStore(&arrivals[tick.neuron_count + other], 0);
    atomicStore(&dispatch_args[other * 3u], 1u);
    atomicMax(&dispatch_args[tick.active_slot * 3u], (tick.event_count + 255u) / 256u);
  }
  let fired = select(spike_ring[tick.ring_slot * tick.neuron_count + source], states[source].spikes, tick.delay_steps == 0u);
  if (fired == 0u || states[source].silenced != 0u || row_ptr[source] == row_ptr[source + 1u]) {
    return;
  }
  let index = u32(atomicAdd(&arrivals[tick.neuron_count + tick.active_slot], 1));
  atomicStore(&arrivals[tick.neuron_count + 2u + tick.active_slot * tick.neuron_count + index], i32(source));
  atomicMax(&dispatch_args[tick.active_slot * 3u], min(index + 1u, MAX_SOURCE_GROUPS));
}

fn decay_threshold(neuron: u32) {
  let remaining = max(states[neuron].refractory - 1, 0);
  if (states[neuron].refractory != remaining) {
    states[neuron].refractory = remaining;
  }
  if (remaining == 0) {
    let old_conductance = states[neuron].conductance;
    if (old_conductance != 0.0 || states[neuron].voltage != tick.resting_mv) {
      states[neuron].voltage = tick.resting_mv
        + tick.membrane_decay * (states[neuron].voltage - tick.resting_mv)
        + tick.coupling * old_conductance;
      states[neuron].conductance = tick.synapse_decay * old_conductance;
    }
  }
  let fired = remaining == 0 && states[neuron].voltage > tick.threshold_mv;
  states[neuron].spikes = select(0u, 1u, fired);
  if (fired) {
    states[neuron].spike_count += 1u;
  }
}

@compute @workgroup_size(256)
fn initial_decay(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let neuron = global_id.x;
  if (neuron < tick.neuron_count) {
    decay_threshold(neuron);
    queue_source(neuron);
  }
}

@compute @workgroup_size(256)
fn reset_decay(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let neuron = global_id.x;
  if (neuron >= tick.neuron_count) {
    return;
  }
  let arrival = atomicLoad(&arrivals[neuron]);
  if (arrival != 0) {
    atomicStore(&arrivals[neuron], 0);
    states[neuron].conductance += f32(arrival) * tick.synapse_weight_mv;
  }
  let previous_spike = states[neuron].spikes;
  let previous_slot = (tick.ring_slot + tick.ring_size - 1u) % tick.ring_size;
  spike_ring[previous_slot * tick.neuron_count + neuron] = previous_spike;
  if (previous_spike != 0u) {
    states[neuron].voltage = tick.reset_mv;
    states[neuron].conductance = 0.0;
    states[neuron].refractory = states[neuron].refractory_length;
  }
  decay_threshold(neuron);
  queue_source(neuron);
}

var<workgroup> source_count: u32;
var<workgroup> edges_begin: u32;
var<workgroup> edges_end: u32;

@compute @workgroup_size(256)
fn propagate(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>,
) {
  let event_index = global_id.x;
  if (event_index < tick.event_count) {
    let packed = events[tick.target_count + tick.event_offset + event_index];
    let destination = events[packed & 0x00ffffffu];
    states[destination].voltage += f32(packed >> 24u) * tick.external_weight_mv;
  }
  if (local_id.x == 0u) {
    source_count = u32(atomicLoad(&arrivals[tick.neuron_count + tick.active_slot]));
  }
  let count = workgroupUniformLoad(&source_count);
  for (var index = group_id.x; index < count; index += MAX_SOURCE_GROUPS) {
    if (local_id.x == 0u) {
      let source = u32(atomicLoad(&arrivals[tick.neuron_count + 2u + tick.active_slot * tick.neuron_count + index]));
      edges_begin = row_ptr[source];
      edges_end = row_ptr[source + 1u];
    }
    let begin = workgroupUniformLoad(&edges_begin);
    let end = workgroupUniformLoad(&edges_end);
    for (var edge = begin + local_id.x; edge < end; edge += SOURCE_LANES) {
      let packed = signed_counts[edge / 2u];
      let signed_count = bitcast<i32>(packed << ((1u - (edge & 1u)) * 16u)) >> 16u;
      atomicAdd(&arrivals[destinations[edge]], signed_count);
    }
  }
}

@compute @workgroup_size(256)
fn reset_store(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let neuron = global_id.x;
  if (neuron >= tick.neuron_count) {
    return;
  }
  let arrival = atomicLoad(&arrivals[neuron]);
  if (arrival != 0) {
    atomicStore(&arrivals[neuron], 0);
    states[neuron].conductance += f32(arrival) * tick.synapse_weight_mv;
  }
  let previous_spike = states[neuron].spikes;
  spike_ring[tick.ring_slot * tick.neuron_count + neuron] = previous_spike;
  if (previous_spike != 0u) {
    states[neuron].voltage = tick.reset_mv;
    states[neuron].conductance = 0.0;
    states[neuron].refractory = states[neuron].refractory_length;
  }
}

struct SummaryParams {
  neuron_count: u32,
  probe_count: u32,
  padding0: u32,
  padding1: u32,
}

@group(1) @binding(0) var<storage, read> summary_states: array<State>;
@group(1) @binding(1) var<storage, read> probes: array<u32>;
@group(1) @binding(2) var<storage, read_write> summary: array<u32>;
@group(1) @binding(3) var<uniform> summary_params: SummaryParams;

@compute @workgroup_size(256)
fn gather_probes(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let probe = global_id.x;
  if (probe < summary_params.probe_count) {
    summary[probe] = summary_states[probes[probe]].spike_count;
  }
}

var<workgroup> low_counts: array<u32, 256>;
var<workgroup> high_counts: array<u32, 256>;
var<workgroup> active_counts: array<u32, 256>;
var<workgroup> voltage_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce_telemetry(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
  let neuron = global_id.x;
  let local = local_id.x;
  if (neuron < summary_params.neuron_count) {
    let count = summary_states[neuron].spike_count;
    low_counts[local] = count & 0xffffu;
    high_counts[local] = count >> 16u;
    active_counts[local] = select(0u, 1u, count != 0u);
    voltage_sums[local] = summary_states[neuron].voltage;
  } else {
    low_counts[local] = 0u;
    high_counts[local] = 0u;
    active_counts[local] = 0u;
    voltage_sums[local] = 0.0;
  }
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (local < stride) {
      low_counts[local] += low_counts[local + stride];
      high_counts[local] += high_counts[local + stride];
      active_counts[local] += active_counts[local + stride];
      voltage_sums[local] += voltage_sums[local + stride];
    }
    workgroupBarrier();
  }
  if (local == 0u) {
    let offset = summary_params.probe_count + workgroup_id.x * 4u;
    summary[offset] = low_counts[0];
    summary[offset + 1u] = high_counts[0];
    summary[offset + 2u] = active_counts[0];
    summary[offset + 3u] = bitcast<u32>(voltage_sums[0]);
  }
}
