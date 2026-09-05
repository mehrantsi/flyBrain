from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from flybrain.connectome import PackedConnectome
from flybrain.protocols import RIGHT_SUGAR_GRN_IDS
from flybrain.runner import run_sugar_experiment


def test_sugar_run_writes_auditable_results(tmp_path: Path) -> None:
    connectome = PackedConnectome.from_arrays(
        neuron_ids=[RIGHT_SUGAR_GRN_IDS[0]],
        row_ptr=[0, 0],
        destinations=[],
        signed_counts=[],
        manifest={"materialization": "test", "array_sha256": {"source": "digest"}},
    )
    output = tmp_path / "run"

    returned = run_sugar_experiment(
        connectome,
        output,
        duration_ms=1.0,
        rate_hz=10_000.0,
        seed=7,
        propagation="scatter",
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert returned == manifest
    assert manifest["steps"] == 10
    assert manifest["stimulus_target_count"] == 1
    assert manifest["missing_stimulus_ids"] == list(RIGHT_SUGAR_GRN_IDS[1:])
    assert manifest["total_spikes"] == int(np.load(output / "spike_counts.npy").sum())
    np.testing.assert_array_equal(
        np.load(output / "neuron_ids.npy"),
        np.array([RIGHT_SUGAR_GRN_IDS[0]], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        np.load(output / "firing_rates_hz.npy"),
        np.load(output / "spike_counts.npy") * 1000.0,
    )
    for name, digest in manifest["output_array_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_sugar_run_does_not_replace_an_existing_output(tmp_path: Path) -> None:
    connectome = PackedConnectome.from_arrays(
        neuron_ids=[RIGHT_SUGAR_GRN_IDS[0]],
        row_ptr=[0, 0],
        destinations=[],
        signed_counts=[],
    )
    output = tmp_path / "run"
    output.mkdir()

    with pytest.raises(FileExistsError):
        run_sugar_experiment(
            connectome,
            output,
            duration_ms=0.1,
            propagation="scatter",
        )
