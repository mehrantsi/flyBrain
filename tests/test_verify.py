from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flybrain.verify import audit_pack


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pack(path: Path) -> None:
    path.mkdir()
    arrays = {
        "neuron_ids.npy": np.array([10, 20], dtype=np.uint64),
        "row_ptr.npy": np.array([0, 1, 1], dtype=np.uint32),
        "destinations.npy": np.array([1], dtype=np.uint32),
        "signed_counts.npy": np.array([-3], dtype=np.int16),
    }
    for name, array in arrays.items():
        np.save(path / name, array)
    manifest = {
        "materialization": "test",
        "neuron_count": 2,
        "edge_count": 1,
        "contact_sum": 3,
        "excitatory_edge_count": 0,
        "inhibitory_edge_count": 1,
        "array_sha256": {name: _sha256(path / name) for name in arrays},
    }
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_audit_pack_recomputes_counts_and_hashes(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    _write_pack(pack)

    audit = audit_pack(pack)

    assert audit.neuron_count == 2
    assert audit.contact_sum == 3
    assert audit.maximum_out_degree == 1
    assert audit.array_hashes_verified


def test_audit_pack_rejects_tampered_array(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    _write_pack(pack)
    np.save(pack / "signed_counts.npy", np.array([-2], dtype=np.int16))

    with pytest.raises(ValueError, match="contact_sum|SHA256"):
        audit_pack(pack)
