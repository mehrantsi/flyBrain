from __future__ import annotations

import pytest

from flybrain.connectome import PackedConnectome
from flybrain.protocols import indices_for_flywire_ids


def test_indices_for_flywire_ids_preserve_requested_order() -> None:
    connectome = PackedConnectome.from_arrays(
        neuron_ids=[30, 10, 20],
        row_ptr=[0, 0, 0, 0],
        destinations=[],
        signed_counts=[],
    )

    assert indices_for_flywire_ids(connectome, [10, 30]).tolist() == [1, 0]

    with pytest.raises(KeyError, match="absent"):
        indices_for_flywire_ids(connectome, [99])

    assert indices_for_flywire_ids(connectome, [99, 20], allow_missing=True).tolist() == [2]
