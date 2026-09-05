from __future__ import annotations

import numpy as np
import pytest

from flybrain.connectome import PackedConnectome


def test_connectome_validates_csr_shape() -> None:
    connectome = PackedConnectome.from_arrays(
        neuron_ids=[10, 20, 30],
        row_ptr=[0, 2, 2, 3],
        destinations=[1, 2, 0],
        signed_counts=[2, -1, 3],
    )

    assert connectome.neuron_count == 3
    assert connectome.edge_count == 3
    np.testing.assert_array_equal(connectome.row_ptr, [0, 2, 2, 3])


def test_connectome_rejects_out_of_range_destination() -> None:
    with pytest.raises(ValueError, match="outside"):
        PackedConnectome.from_arrays(
            neuron_ids=[10], row_ptr=[0, 1], destinations=[1], signed_counts=[1]
        )
