from __future__ import annotations

import hashlib
import json
import shutil
from itertools import pairwise
from pathlib import Path

import pytest

from tools.build_male_cns_io import build_male_cns_io

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = (
    ROOT / "work/upstream/male-cns/v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather"
)
NEUROTRANSMITTERS = (
    ROOT / "work/upstream/male-cns/v1.0/body-neurotransmitters-male-cns-v1.0.feather"
)
CONNECTIVITY = (
    ROOT / "work/upstream/male-cns/v1.0/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)
PACK = ROOT / "outputs/packs/male_cns_v1"


def _build(tmp_path: Path) -> tuple[dict, dict]:
    return build_male_cns_io(
        ANNOTATIONS,
        NEUROTRANSMITTERS,
        CONNECTIVITY,
        PACK,
        tmp_path / "male_cns_v1_neural_io.json",
        tmp_path / "male_cns_v1_neural_io_evidence.json",
    )


def test_builder_binds_real_groups_and_routes_to_the_male_cns_pack(tmp_path: Path) -> None:
    artifact, evidence = _build(tmp_path)

    assert artifact["schema_version"] == 1
    assert artifact["dataset"]["materialization"] == "male-cns-v1.0-superclass-non-null-known-nt"
    assert set(artifact["dataset"]["pack_array_sha256"]) == {
        "destinations.npy",
        "neuron_ids.npy",
        "row_ptr.npy",
        "signed_counts.npy",
    }
    assert len(artifact["groups"]["taste_sugar"]["root_ids"]) == 85
    assert artifact["groups"]["feeding_mn9"]["root_ids"] == [10331]
    assert len(artifact["groups"]["olfaction_left"]["root_ids"]) == 798
    assert len(artifact["groups"]["olfaction_right"]["root_ids"]) == 1247
    assert len(artifact["groups"]["motor_flight_power_left"]["root_ids"]) == 12
    assert len(artifact["groups"]["motor_flight_power_right"]["root_ids"]) == 12
    assert len(artifact["groups"]["motor_flight_steering_left"]["root_ids"]) == 3
    assert len(artifact["groups"]["motor_flight_steering_right"]["root_ids"]) == 3
    assert artifact["food_olfaction"]["annotation_field"] == "type"
    assert not any(
        "R1-R6" in row.get("type_counts", {}) for row in evidence["group_census"].values()
    )
    assert any(route["name"] == "taste_to_feeding_mn9" for route in evidence["routes"])
    assert any(route["name"] == "visual_loom_to_landing_left" for route in evidence["routes"])


def test_route_witnesses_are_edges_in_the_audited_csr(tmp_path: Path) -> None:
    _, evidence = _build(tmp_path)
    import numpy as np

    ids = np.load(PACK / "neuron_ids.npy")
    row_ptr = np.load(PACK / "row_ptr.npy")
    destinations = np.load(PACK / "destinations.npy")
    index_by_id = {int(root_id): index for index, root_id in enumerate(ids)}
    for route in evidence["routes"]:
        path = route["path_root_ids"]
        assert len(route["edge_signed_counts"]) == len(path) - 1
        for pre_id, post_id in pairwise(path):
            pre_index = index_by_id[pre_id]
            post_index = index_by_id[post_id]
            start, end = int(row_ptr[pre_index]), int(row_ptr[pre_index + 1])
            assert int(post_index) in destinations[start:end]


def test_builder_rejects_a_source_that_is_not_hash_bound_to_the_pack(tmp_path: Path) -> None:
    tampered = tmp_path / "annotations.feather"
    shutil.copyfile(ANNOTATIONS, tampered)
    payload = bytearray(tampered.read_bytes())
    payload[-1] ^= 1
    tampered.write_bytes(payload)

    with pytest.raises(ValueError, match="source hash"):
        build_male_cns_io(
            tampered,
            NEUROTRANSMITTERS,
            CONNECTIVITY,
            PACK,
            tmp_path / "artifact.json",
            tmp_path / "evidence.json",
        )


def test_generated_json_is_deterministic_and_records_array_hashes(tmp_path: Path) -> None:
    artifact, _ = _build(tmp_path)
    output = tmp_path / "male_cns_v1_neural_io.json"
    decoded = json.loads(output.read_text(encoding="utf-8"))
    assert decoded == artifact
    for name, expected in artifact["dataset"]["pack_array_sha256"].items():
        actual = hashlib.sha256((PACK / name).read_bytes()).hexdigest()
        assert actual == expected
