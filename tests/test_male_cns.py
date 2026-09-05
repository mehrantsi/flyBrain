from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from pyarrow import feather

from flybrain.connectome import PackedConnectome
from flybrain.male_cns import import_male_cns


def _write_sources(
    root: Path,
    *,
    annotations: list[tuple[int, str | None, str | None]],
    neurotransmitters: list[tuple[int, str]],
    edges: list[tuple[int, int, int]],
) -> tuple[Path, Path, Path]:
    annotation_path = root / "annotations.feather"
    nt_path = root / "neurotransmitters.feather"
    connectivity_path = root / "connectivity.feather"
    feather.write_feather(
        pa.table(
            {
                "bodyId": [row[0] for row in annotations],
                "superclass": [row[1] for row in annotations],
                "status": [row[2] for row in annotations],
            }
        ),
        annotation_path,
    )
    feather.write_feather(
        pa.table(
            {
                "body": [row[0] for row in neurotransmitters],
                "consensus_nt": [row[1] for row in neurotransmitters],
            }
        ),
        nt_path,
    )
    feather.write_feather(
        pa.table(
            {
                "body_pre": [row[0] for row in edges],
                "body_post": [row[1] for row in edges],
                "weight": [row[2] for row in edges],
            }
        ),
        connectivity_path,
    )
    return annotation_path, nt_path, connectivity_path


def test_import_male_cns_filters_nodes_and_unknown_nt_edges(tmp_path: Path) -> None:
    annotation, nt, connectivity = _write_sources(
        tmp_path,
        annotations=[
            (30, "vnc_motor", "Traced"),
            (10, "vnc_sensory", "Traced"),
            (20, None, "Unimportant"),
            (40, "vnc_intrinsic", None),
        ],
        neurotransmitters=[(10, "ACh"), (30, "GABA"), (40, "unclear")],
        edges=[
            (30, 10, 2),
            (10, 30, 3),
            (10, 40, 4),
            (30, 40, 1),
            (40, 10, 5),
            (10, 20, 6),
            (20, 10, 7),
        ],
    )
    output = tmp_path / "pack"

    manifest = import_male_cns(annotation, nt, connectivity, output)

    np.testing.assert_array_equal(np.load(output / "neuron_ids.npy"), [10, 30, 40])
    np.testing.assert_array_equal(np.load(output / "row_ptr.npy"), [0, 2, 4, 4])
    np.testing.assert_array_equal(np.load(output / "destinations.npy"), [1, 2, 0, 2])
    np.testing.assert_array_equal(np.load(output / "signed_counts.npy"), [3, 4, -2, -1])
    PackedConnectome.load(output)

    assert manifest["neuron_count"] == 3
    assert manifest["edge_count"] == 4
    assert manifest["contact_sum"] == 10
    assert manifest["counts"]["raw_connectivity_rows"] == 7
    assert manifest["counts"]["selected_endpoint_edge_rows"] == 5
    assert manifest["counts"]["omitted_unknown_nt_edges"] == 1
    assert manifest["counts"]["omitted_unknown_nt_contact_sum"] == 5
    assert manifest["neurotransmitter_policy"]["confidence_threshold"] is None
    assert manifest["selection"]["predicate"] == "annotations.superclass != null"
    assert manifest["selection"]["selected_neuron_count"] == 3
    assert json.loads((output / "manifest.json").read_text()) == manifest
    for name, digest in manifest["array_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_import_male_cns_rejects_duplicate_nt_join(tmp_path: Path) -> None:
    annotation, nt, connectivity = _write_sources(
        tmp_path,
        annotations=[(10, "sensory", "Traced")],
        neurotransmitters=[(10, "ACh"), (10, "GABA")],
        edges=[(10, 10, 1)],
    )

    with pytest.raises(ValueError, match="neurotransmitter body values must be unique"):
        import_male_cns(annotation, nt, connectivity, tmp_path / "pack")


@pytest.mark.parametrize("label,sign", [
    ("ACh", 1), ("GABA", -1), ("Glu", -1), ("histamine", 0), ("unclear", 0), ("dopamine", 0),
])
def test_exact_release_transmitter_labels(tmp_path: Path, label: str, sign: int) -> None:
    sources = _write_sources(
        tmp_path,
        annotations=[(10, "vnc_intrinsic", "Traced"), (20, "vnc_motor", "Traced")],
        neurotransmitters=[(10, label), (20, "unclear")],
        edges=[(10, 20, 3)],
    )
    output = tmp_path / "pack"
    manifest = import_male_cns(*sources, output)
    np.testing.assert_array_equal(np.load(output / "signed_counts.npy"), [sign * 3] if sign else [])
    assert manifest["neuron_count"] == 2
    assert manifest["counts"]["omitted_unknown_nt_edges"] == int(sign == 0)


def test_import_male_cns_rejects_duplicate_selected_edge(tmp_path: Path) -> None:
    annotation, nt, connectivity = _write_sources(
        tmp_path,
        annotations=[(10, "sensory", "Traced"), (20, "motor", "Traced")],
        neurotransmitters=[(10, "ACh"), (20, "GABA")],
        edges=[(10, 20, 1), (10, 20, 2)],
    )

    with pytest.raises(ValueError, match="duplicate presynaptic/postsynaptic pair"):
        import_male_cns(annotation, nt, connectivity, tmp_path / "pack")


def test_import_male_cns_rejects_out_of_int16_weight(tmp_path: Path) -> None:
    annotation, nt, connectivity = _write_sources(
        tmp_path,
        annotations=[(10, "sensory", "Traced"), (20, "motor", "Traced")],
        neurotransmitters=[(10, "ACh"), (20, "GABA")],
        edges=[(10, 20, 32768)],
    )

    with pytest.raises(ValueError, match="signed int16"):
        import_male_cns(annotation, nt, connectivity, tmp_path / "pack")
