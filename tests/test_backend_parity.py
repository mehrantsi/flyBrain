from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from flybrain.connectome import PackedConnectome
from flybrain.mlx_engine import MLXEngine
from flybrain.parameters import ModelParameters
from flybrain.reference import ReferenceSimulator


def recurrent_connectome() -> PackedConnectome:
    return PackedConnectome.from_arrays(
        neuron_ids=[10, 20, 30, 40],
        row_ptr=[0, 2, 3, 4, 4],
        destinations=[1, 2, 3, 3],
        signed_counts=[5, 2, 3, -2],
    )


@pytest.mark.parametrize("propagation", ["scatter", "metal"])
def test_mlx_backends_track_float64_reference(propagation: str) -> None:
    parameters = ModelParameters(delay_ms=0.2, refractory_ms=0.3)
    connectome = recurrent_connectome()
    reference = ReferenceSimulator(connectome, parameters, activated=[0])
    engine = MLXEngine(
        connectome,
        parameters,
        propagation=propagation,
        zero_refractory=[0],
    )
    events = np.zeros((80, connectome.neuron_count), dtype=np.int32)
    events[[0, 8, 15, 31, 47, 63], 0] = 1

    for external in events:
        expected_spikes = reference.step(external)
        actual = engine.step(external, record_state=True)

        np.testing.assert_array_equal(actual.spikes, expected_spikes)
        np.testing.assert_allclose(actual.voltage_mv, reference.v, rtol=2e-6, atol=2e-5)
        np.testing.assert_allclose(actual.conductance_mv, reference.g, rtol=2e-6, atol=2e-5)
