from __future__ import annotations

import numpy as np
import pytest

from flybrain.connectome import PackedConnectome
from flybrain.parameters import ModelParameters
from flybrain.reference import ReferenceSimulator


def _network(
    neuron_count: int,
    *,
    edges: list[tuple[int, int, int]] | None = None,
    delay_ms: float = 0.0,
    refractory_ms: float = 0.2,
) -> tuple[PackedConnectome, ModelParameters]:
    edges = [] if edges is None else edges
    by_source = [[] for _ in range(neuron_count)]
    for source, destination, count in edges:
        by_source[source].append((destination, count))
    destinations: list[int] = []
    signed_counts: list[int] = []
    row_ptr = [0]
    for source_edges in by_source:
        destinations.extend(destination for destination, _ in source_edges)
        signed_counts.extend(count for _, count in source_edges)
        row_ptr.append(len(destinations))
    connectome = PackedConnectome.from_arrays(
        neuron_ids=np.arange(neuron_count),
        row_ptr=row_ptr,
        destinations=destinations,
        signed_counts=signed_counts,
    )
    return connectome, ModelParameters(delay_ms=delay_ms, refractory_ms=refractory_ms)


def test_state_update_uses_exact_two_exponential_solution() -> None:
    connectome, parameters = _network(1)
    simulator = ReferenceSimulator(connectome, parameters, initial_v=[-50.0], initial_g=[1.0])

    simulator.step()

    expected_g = np.exp(-parameters.dt_ms / parameters.synapse_tau_ms)
    expected_v = (
        parameters.resting_mv
        + (-50.0 - parameters.resting_mv) * np.exp(-parameters.dt_ms / parameters.membrane_tau_ms)
        + parameters.membrane_synapse_coupling
    )
    assert simulator.g[0] == pytest.approx(expected_g)
    assert simulator.v[0] == pytest.approx(expected_v)


def test_threshold_is_strictly_greater_than_threshold() -> None:
    connectome, parameters = _network(1)
    at_threshold = ReferenceSimulator(connectome, parameters, initial_v=[parameters.threshold_mv])
    above_threshold = ReferenceSimulator(
        connectome, parameters, initial_v=[parameters.threshold_mv + 0.5]
    )

    assert not at_threshold.step()[0]
    assert above_threshold.step()[0]


def test_synaptic_event_arrives_after_configured_delay() -> None:
    connectome, parameters = _network(2, edges=[(0, 1, 1)], delay_ms=0.2)
    simulator = ReferenceSimulator(connectome, parameters, initial_v=[-44.0, -52.0])

    assert simulator.step()[0]
    assert simulator.g[1] == pytest.approx(0.0)
    simulator.step()
    assert simulator.g[1] == pytest.approx(0.0)
    simulator.step()
    assert simulator.g[1] == pytest.approx(parameters.synapse_weight_mv)


def test_refractory_period_blocks_thresholding() -> None:
    connectome, parameters = _network(1, refractory_ms=0.2)
    simulator = ReferenceSimulator(
        connectome,
        parameters,
        initial_v=[-44.0],
        external_weight_mv=20.0,
    )

    spikes = [bool(simulator.step(external_counts=[1.0])[0])]
    spikes.extend(bool(simulator.step(external_counts=[1.0])[0]) for _ in range(3))

    assert spikes == [True, False, True, False]


def test_negative_signed_count_is_inhibitory() -> None:
    connectome, parameters = _network(2, edges=[(0, 1, -2)], delay_ms=0.1)
    simulator = ReferenceSimulator(connectome, parameters, initial_v=[-44.0, -52.0])

    simulator.step()
    simulator.step()

    assert simulator.g[1] < 0.0
    assert simulator.g[1] == pytest.approx(-2 * parameters.synapse_weight_mv)


def test_silencing_disables_outbound_events_but_not_source_spike() -> None:
    connectome, parameters = _network(2, edges=[(0, 1, 1)], delay_ms=0.1)
    simulator = ReferenceSimulator(
        connectome,
        parameters,
        initial_v=[-44.0, -52.0],
        silenced_sources=[0],
    )

    assert simulator.step()[0]
    simulator.step()

    assert simulator.g[1] == pytest.approx(0.0)


def test_external_event_is_applied_after_threshold_and_spikes_next_step() -> None:
    connectome, parameters = _network(1)
    simulator = ReferenceSimulator(
        connectome,
        parameters,
        activated=[0],
        external_weight_mv=20.0,
    )

    result = simulator.run(steps=2, external_events={0: {0: 1}}, record=True)

    assert result is not None
    assert not result.spikes[0, 0]
    assert result.spikes[1, 0]
    assert result.spike_trains[0].tolist() == [pytest.approx(0.1)]
