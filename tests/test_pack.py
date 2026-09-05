from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flybrain.pack import pack_connectome

_COLUMNS = [
    "Presynaptic_ID",
    "Postsynaptic_ID",
    "Presynaptic_Index",
    "Postsynaptic_Index",
    "Connectivity",
    "Excitatory",
    "Excitatory x Connectivity",
]


def _write_inputs(tmp_path: Path, *, bad_presynaptic_id: bool = False) -> tuple[Path, Path]:
    completeness = tmp_path / "completeness.csv"
    completeness.write_text(",Completed\n101,True\n202,True\n303,True\n", encoding="utf-8")

    rows = [
        (202, 101, 1, 0, 2, -1, -2),
        (101, 303, 0, 2, 3, 1, 3),
        (101, 202, 0, 1, 1, 1, 1),
    ]
    if bad_presynaptic_id:
        rows[0] = (999, 101, 1, 0, 2, -1, -2)
    table = pa.table({name: [row[index] for row in rows] for index, name in enumerate(_COLUMNS)})
    connectivity = tmp_path / "connectivity.parquet"
    pq.write_table(table, connectivity)
    return completeness, connectivity


def test_pack_connectome_materializes_stable_csr_and_manifest(tmp_path: Path) -> None:
    completeness, connectivity = _write_inputs(tmp_path)
    output = tmp_path / "packed"

    manifest = pack_connectome(completeness, connectivity, output, materialization="test")

    np.testing.assert_array_equal(
        np.load(output / "neuron_ids.npy"),
        np.array([101, 202, 303], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        np.load(output / "row_ptr.npy"),
        np.array([0, 2, 3, 3], dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        np.load(output / "destinations.npy"),
        np.array([2, 1, 0], dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        np.load(output / "signed_counts.npy"),
        np.array([3, 1, -2], dtype=np.int16),
    )

    assert manifest["schema_version"] == 1
    assert manifest["materialization"] == "test"
    assert manifest["neuron_count"] == 3
    assert manifest["edge_count"] == 3
    assert manifest["contact_sum"] == 6
    assert manifest["excitatory_edge_count"] == 2
    assert manifest["inhibitory_edge_count"] == 1
    assert json.loads((output / "manifest.json").read_text()) == manifest

    for name, digest in manifest["array_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_pack_connectome_rejects_corrupt_id_index_mapping(tmp_path: Path) -> None:
    completeness, connectivity = _write_inputs(tmp_path, bad_presynaptic_id=True)

    with pytest.raises(ValueError, match="Presynaptic_ID.*Presynaptic_Index"):
        pack_connectome(completeness, connectivity, tmp_path / "packed", "test")


def test_pack_connectome_does_not_replace_existing_output(tmp_path: Path) -> None:
    completeness, connectivity = _write_inputs(tmp_path)
    output = tmp_path / "packed"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        pack_connectome(completeness, connectivity, output, "test")

    assert marker.read_text(encoding="utf-8") == "keep"
