from __future__ import annotations

import numpy as np
import pytest

brian2 = pytest.importorskip("brian2")
from brian2 import Clock, Network, NeuronGroup, SpikeMonitor, StateMonitor, ms, mV

from flybrain.brian_oracle import (
    DEFAULT_RECORD_SLOTS,
    RESOLVED_RESET,
    SYNAPSES_SCHEDULE_SLOT,
    UPSTREAM_RESET,
    Brian2Oracle,
    ScheduledEvent,
)
from flybrain.connectome import PackedConnectome
from flybrain.parameters import ModelParameters
from flybrain.reference import ReferenceSimulator


def _one_edge() -> PackedConnectome:
    return PackedConnectome.from_arrays(
        neuron_ids=[10, 20],
        row_ptr=[0, 1, 1],
        destinations=[1],
        signed_counts=[1],
    )


def test_explicit_events_and_state_boundaries_are_tick_aligned() -> None:
    parameters = ModelParameters()
    oracle = Brian2Oracle(
        _one_edge(),
        parameters,
        event_schedule=[ScheduledEvent(neuron_index=0, time_ms=0.0)],
    )

    assert oracle.synapses.pre.when == SYNAPSES_SCHEDULE_SLOT
    assert oracle.event_synapses is not None
    assert oracle.event_synapses.pre.when == SYNAPSES_SCHEDULE_SLOT
    assert oracle.network.schedule == list(DEFAULT_RECORD_SLOTS)

    trace = oracle.run(3.0)
    expected_times = np.arange(30, dtype=float) * parameters.dt_ms
    np.testing.assert_allclose(trace.times_ms, expected_times, atol=1e-12)

    # The source event occurs at t=0, but the 1.8 ms propagation delay means
    # that the pre-event is visible only in the synapses slot at tick 18.
    assert trace.g_by_slot_mv["groups"][18, 1] == pytest.approx(0.0, abs=1e-12)
    assert trace.g_by_slot_mv["synapses"][17, 1] == pytest.approx(0.0, abs=1e-12)
    assert trace.g_by_slot_mv["synapses"][18, 1] == pytest.approx(parameters.synapse_weight_mv)
    assert trace.g_by_slot_mv["end"][18, 1] == pytest.approx(parameters.synapse_weight_mv)
    assert trace.g_by_slot_mv["synapses"][19, 1] == pytest.approx(
        parameters.synapse_weight_mv * parameters.synapse_decay
    )
    assert trace.spike_indices.size == 0


def test_source_silencing_applies_to_internal_and_deterministic_synapses() -> None:
    oracle = Brian2Oracle(
        _one_edge(),
        event_schedule=[ScheduledEvent(0, 0.0)],
        silenced_sources=[0],
    )

    np.testing.assert_allclose(np.asarray(oracle.synapses.w / mV), [0.0])
    assert oracle.event_synapses is not None
    np.testing.assert_allclose(np.asarray(oracle.event_synapses.w / mV), [0.0])

    trace = oracle.run(3.0)
    for values in trace.g_by_slot_mv.values():
        np.testing.assert_allclose(values[:, 1], 0.0, atol=1e-12)


def test_zero_refractory_is_a_per_neuron_override() -> None:
    oracle = Brian2Oracle(_one_edge(), zero_refractory=[1])

    np.testing.assert_allclose(np.asarray(oracle.neurons.rfc[:] / ms), [2.2, 0.0])


def _run_reset(reset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # A deliberately direct Brian2 construction keeps this test focused on
    # the upstream reset typo rather than on the adapter's resolved constant.
    brian2.start_scope()
    clock = Clock(dt=0.1 * ms)
    equations = """
    dv/dt = (resting - v + g) / membrane_tau : volt (unless refractory)
    dg/dt = -g / synapse_tau : volt (unless refractory)
    rfc : second
    """
    neurons = NeuronGroup(
        1,
        equations,
        method="linear",
        threshold="v > threshold",
        reset=reset,
        refractory="rfc",
        namespace={
            "resting": -52 * mV,
            "reset_mv": -52 * mV,
            "threshold": -45 * mV,
            "membrane_tau": 20 * ms,
            "synapse_tau": 5 * ms,
        },
        clock=clock,
    )
    neurons.v = -45 * mV
    neurons.g = 10 * mV
    neurons.rfc = 0 * ms
    spikes = SpikeMonitor(neurons)
    state = StateMonitor(neurons, ("v", "g"), record=True, when="end", order=100)
    Network(neurons, spikes, state).run(0.2 * ms)
    return (
        np.asarray(state.v / mV).copy(),
        np.asarray(state.g / mV).copy(),
        np.asarray(spikes.t / ms).copy(),
    )


def test_upstream_w_reset_is_a_dead_assignment() -> None:
    upstream = _run_reset(UPSTREAM_RESET)
    resolved = _run_reset(RESOLVED_RESET)

    for upstream_values, resolved_values in zip(upstream, resolved):
        np.testing.assert_allclose(upstream_values, resolved_values, atol=1e-12)


def test_numpy_reference_matches_brian_tick_for_tick() -> None:
    connectome = PackedConnectome.from_arrays(
        neuron_ids=[10, 20],
        row_ptr=[0, 1, 2],
        destinations=[1, 0],
        signed_counts=[50, -3],
    )
    parameters = ModelParameters(delay_ms=0.2, refractory_ms=0.3)
    initial_v = np.array([-44.0, -52.0])
    brian = Brian2Oracle(
        connectome,
        parameters,
        initial_v_mv=initial_v,
        record_slots=("end",),
    ).run(1.5)
    reference = ReferenceSimulator(
        connectome,
        parameters,
        initial_v=initial_v,
    ).run(steps=15, record=True, record_state=True)

    assert reference is not None
    np.testing.assert_allclose(reference.times_ms, brian.times_ms, atol=1e-12)
    np.testing.assert_allclose(reference.voltage, brian.v_mv, rtol=0, atol=1e-11)
    np.testing.assert_allclose(reference.conductance, brian.g_mv, rtol=0, atol=1e-11)
    expected_step, expected_neuron = np.nonzero(reference.spikes)
    np.testing.assert_array_equal(expected_neuron, brian.spike_indices)
    np.testing.assert_allclose(
        expected_step * parameters.dt_ms,
        brian.spike_times_ms,
        atol=1e-12,
    )
