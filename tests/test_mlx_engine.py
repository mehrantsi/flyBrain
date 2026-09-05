from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from flybrain.connectome import PackedConnectome
from flybrain.mlx_engine import MLXEngine
from flybrain.parameters import ModelParameters


def tiny_connectome() -> PackedConnectome:
    return PackedConnectome.from_arrays(
        neuron_ids=[10, 20],
        row_ptr=[0, 1, 1],
        destinations=[1],
        signed_counts=[4],
    )


def test_scatter_external_input_spikes_on_following_step() -> None:
    engine = MLXEngine(
        tiny_connectome(),
        propagation="scatter",
        zero_refractory=[0],
    )
    events = np.zeros((2, 2), dtype=np.int32)
    events[0, 0] = 1

    result = engine.run(2, events, record_state=True)

    assert not result[0].spikes.any()
    assert result[1].spikes.tolist() == [True, False]


def test_scatter_delivers_after_configured_delay() -> None:
    parameters = ModelParameters(delay_ms=0.2)
    engine = MLXEngine(
        tiny_connectome(),
        parameters,
        propagation="scatter",
        zero_refractory=[0],
    )
    events = np.zeros((4, 2), dtype=np.int32)
    events[0, 0] = 1

    result = engine.run(4, events, record_state=True)

    assert result[1].spikes.tolist() == [True, False]
    assert result[2].conductance_mv[1] == pytest.approx(0.0)
    assert result[3].conductance_mv[1] == pytest.approx(1.1)


def test_metal_event_kernel_matches_scatter_backend() -> None:
    parameters = ModelParameters(delay_ms=0.2)
    events = np.zeros((5, 2), dtype=np.int32)
    events[0, 0] = 1
    scatter = MLXEngine(
        tiny_connectome(),
        parameters,
        propagation="scatter",
        zero_refractory=[0],
    ).run(5, events, record_state=True)
    metal = MLXEngine(
        tiny_connectome(),
        parameters,
        propagation="metal",
        zero_refractory=[0],
    ).run(5, events, record_state=True)

    for expected, actual in zip(scatter, metal, strict=True):
        np.testing.assert_array_equal(actual.spikes, expected.spikes)
        np.testing.assert_allclose(actual.voltage_mv, expected.voltage_mv)
        np.testing.assert_allclose(actual.conductance_mv, expected.conductance_mv)
