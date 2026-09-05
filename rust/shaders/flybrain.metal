#include <metal_stdlib>

using namespace metal;

kernel void decay_threshold(
    device float* voltage [[buffer(0)]],
    device float* conductance [[buffer(1)]],
    device int* refractory_remaining [[buffer(2)]],
    device uchar* spikes [[buffer(3)]],
    device uint* spike_counts [[buffer(4)]],
    constant uint& neuron_count [[buffer(5)]],
    constant float& resting_mv [[buffer(6)]],
    constant float& threshold_mv [[buffer(7)]],
    constant float& membrane_decay [[buffer(8)]],
    constant float& synapse_decay [[buffer(9)]],
    constant float& coupling [[buffer(10)]],
    uint neuron [[thread_position_in_grid]]) {
    if (neuron >= neuron_count) {
        return;
    }

    int remaining = max(refractory_remaining[neuron] - 1, 0);
    refractory_remaining[neuron] = remaining;
    bool can_update = remaining == 0;
    float old_g = conductance[neuron];
    if (can_update) {
        voltage[neuron] = resting_mv
            + membrane_decay * (voltage[neuron] - resting_mv)
            + coupling * old_g;
        conductance[neuron] = synapse_decay * old_g;
    }

    bool fired = can_update && voltage[neuron] > threshold_mv;
    spikes[neuron] = fired ? 1 : 0;
    if (fired) {
        spike_counts[neuron] += 1;
    }
}

kernel void propagate_csr(
    device const uint* row_ptr [[buffer(0)]],
    device const uint* destinations [[buffer(1)]],
    device const short* signed_counts [[buffer(2)]],
    device const uchar* delayed_spikes [[buffer(3)]],
    device const uchar* silenced_sources [[buffer(4)]],
    device atomic_int* arrivals [[buffer(5)]],
    constant uint& neuron_count [[buffer(6)]],
    uint source [[thread_position_in_grid]]) {
    if (source >= neuron_count || delayed_spikes[source] == 0 || silenced_sources[source] != 0) {
        return;
    }

    uint begin = row_ptr[source];
    uint end = row_ptr[source + 1];
    for (uint edge = begin; edge < end; ++edge) {
        atomic_fetch_add_explicit(
            &arrivals[destinations[edge]],
            int(signed_counts[edge]),
            memory_order_relaxed);
    }
}

kernel void decay_threshold_propagate_delayed(
    device float* voltage [[buffer(0)]],
    device float* conductance [[buffer(1)]],
    device int* refractory_remaining [[buffer(2)]],
    device uchar* spikes [[buffer(3)]],
    device uint* spike_counts [[buffer(4)]],
    device const uint* row_ptr [[buffer(5)]],
    device const uint* destinations [[buffer(6)]],
    device const short* signed_counts [[buffer(7)]],
    device const uchar* delayed_spikes [[buffer(8)]],
    device const uchar* silenced_sources [[buffer(9)]],
    device atomic_int* arrivals [[buffer(10)]],
    constant uint& neuron_count [[buffer(11)]],
    constant float& resting_mv [[buffer(12)]],
    constant float& threshold_mv [[buffer(13)]],
    constant float& membrane_decay [[buffer(14)]],
    constant float& synapse_decay [[buffer(15)]],
    constant float& coupling [[buffer(16)]],
    uint neuron [[thread_position_in_grid]]) {
    if (neuron >= neuron_count) {
        return;
    }

    int remaining = max(refractory_remaining[neuron] - 1, 0);
    refractory_remaining[neuron] = remaining;
    bool can_update = remaining == 0;
    float old_g = conductance[neuron];
    if (can_update) {
        voltage[neuron] = resting_mv
            + membrane_decay * (voltage[neuron] - resting_mv)
            + coupling * old_g;
        conductance[neuron] = synapse_decay * old_g;
    }

    bool fired = can_update && voltage[neuron] > threshold_mv;
    spikes[neuron] = fired ? 1 : 0;
    if (fired) {
        spike_counts[neuron] += 1;
    }

    if (delayed_spikes[neuron] == 0 || silenced_sources[neuron] != 0) {
        return;
    }
    uint begin = row_ptr[neuron];
    uint end = row_ptr[neuron + 1];
    for (uint edge = begin; edge < end; ++edge) {
        atomic_fetch_add_explicit(
            &arrivals[destinations[edge]],
            int(signed_counts[edge]),
            memory_order_relaxed);
    }
}

kernel void reset_decay_threshold(
    device float* voltage [[buffer(0)]],
    device float* conductance [[buffer(1)]],
    device int* refractory_remaining [[buffer(2)]],
    device const int* refractory_lengths [[buffer(3)]],
    device uchar* spikes [[buffer(4)]],
    device uint* spike_counts [[buffer(5)]],
    device uchar* spike_ring [[buffer(6)]],
    device atomic_int* arrivals [[buffer(7)]],
    constant uint& neuron_count [[buffer(8)]],
    constant uint& previous_ring_slot [[buffer(9)]],
    constant float& reset_mv [[buffer(10)]],
    constant float& synapse_weight_mv [[buffer(11)]],
    constant float& resting_mv [[buffer(12)]],
    constant float& threshold_mv [[buffer(13)]],
    constant float& membrane_decay [[buffer(14)]],
    constant float& synapse_decay [[buffer(15)]],
    constant float& coupling [[buffer(16)]],
    uint neuron [[thread_position_in_grid]]) {
    if (neuron >= neuron_count) {
        return;
    }

    int signed_count = atomic_exchange_explicit(
        &arrivals[neuron], 0, memory_order_relaxed);
    conductance[neuron] += float(signed_count) * synapse_weight_mv;
    uchar previously_fired = spikes[neuron];
    spike_ring[previous_ring_slot * neuron_count + neuron] = previously_fired;
    if (previously_fired != 0) {
        voltage[neuron] = reset_mv;
        conductance[neuron] = 0.0f;
        refractory_remaining[neuron] = refractory_lengths[neuron];
    }

    int remaining = max(refractory_remaining[neuron] - 1, 0);
    refractory_remaining[neuron] = remaining;
    bool can_update = remaining == 0;
    float old_g = conductance[neuron];
    if (can_update) {
        voltage[neuron] = resting_mv
            + membrane_decay * (voltage[neuron] - resting_mv)
            + coupling * old_g;
        conductance[neuron] = synapse_decay * old_g;
    }

    bool fired = can_update && voltage[neuron] > threshold_mv;
    spikes[neuron] = fired ? 1 : 0;
    if (fired) {
        spike_counts[neuron] += 1;
    }
}

kernel void apply_external(
    device float* voltage [[buffer(0)]],
    device const uint* targets [[buffer(1)]],
    device const uchar* event_counts [[buffer(2)]],
    constant uint& event_offset [[buffer(3)]],
    constant uint& target_count [[buffer(4)]],
    constant float& external_weight_mv [[buffer(5)]],
    uint target_offset [[thread_position_in_grid]]) {
    if (target_offset >= target_count) {
        return;
    }

    uchar count = event_counts[event_offset + target_offset];
    if (count != 0) {
        voltage[targets[target_offset]] += float(count) * external_weight_mv;
    }
}

kernel void apply_external_sparse(
    device float* voltage [[buffer(0)]],
    device const uint* targets_by_lane [[buffer(1)]],
    device const uint* event_lanes [[buffer(2)]],
    device const uchar* event_counts [[buffer(3)]],
    constant uint& event_offset [[buffer(4)]],
    constant uint& event_count [[buffer(5)]],
    constant float& external_weight_mv [[buffer(6)]],
    uint event_index [[thread_position_in_grid]]) {
    if (event_index >= event_count) {
        return;
    }

    uint packed_index = event_offset + event_index;
    voltage[targets_by_lane[event_lanes[packed_index]]] +=
        float(event_counts[packed_index]) * external_weight_mv;
}

kernel void reset_store(
    device float* voltage [[buffer(0)]],
    device float* conductance [[buffer(1)]],
    device int* refractory_remaining [[buffer(2)]],
    device const int* refractory_lengths [[buffer(3)]],
    device const uchar* spikes [[buffer(4)]],
    device uchar* spike_ring [[buffer(5)]],
    device atomic_int* arrivals [[buffer(6)]],
    constant uint& neuron_count [[buffer(7)]],
    constant uint& ring_slot [[buffer(8)]],
    constant float& reset_mv [[buffer(9)]],
    constant float& synapse_weight_mv [[buffer(10)]],
    uint neuron [[thread_position_in_grid]]) {
    if (neuron >= neuron_count) {
        return;
    }

    int signed_count = atomic_exchange_explicit(
        &arrivals[neuron], 0, memory_order_relaxed);
    conductance[neuron] += float(signed_count) * synapse_weight_mv;
    uchar fired = spikes[neuron];
    spike_ring[ring_slot * neuron_count + neuron] = fired;
    if (fired != 0) {
        voltage[neuron] = reset_mv;
        conductance[neuron] = 0.0f;
        refractory_remaining[neuron] = refractory_lengths[neuron];
    }
}
