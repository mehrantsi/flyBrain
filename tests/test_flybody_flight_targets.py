from __future__ import annotations

# ruff: noqa: I001

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "neuromechfly"
SOURCE = ROOT / (
    "work/upstream/flybody-data/datasets_flight-imitation/"
    "flight-dataset_saccade-evasion_augmented.hdf5"
)
TARGETS = ASSETS / "flight_system_id_targets_v1.json"


def test_flybody_flight_targets_are_pair_preserving_and_provenanced():
    payload = json.loads(TARGETS.read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert payload["schema"] == "flybrain.flight-system-id-targets"
    assert payload["schema_version"] == 1
    assert payload["source"]["sha256"] == source_hash
    assert payload["dataset"]["trajectory_count"] == 272
    assert payload["dataset"]["pair_count"] == 136
    assert payload["split"]["trajectory_counts"] == {
        "train": 196,
        "validation": 40,
        "test": 36,
    }
    assert payload["target_contract"]["excluded_targets"] == [
        "absolute room altitude",
        "absolute x/y room position",
        "raw qpos/qvel time series",
    ]

    trajectories = {item["trajectory_index"]: item for item in payload["trajectories"]}
    assert len(trajectories) == 272
    for pair in payload["pairing"]["pairs"]:
        original = trajectories[pair["original_index"]]
        reflected = trajectories[pair["reflected_index"]]
        assert original["pair_id"] == reflected["pair_id"] == pair["pair_id"]
        assert original["split"] == reflected["split"] == pair["split"]
        assert original["reflection"] == "original"
        assert reflected["reflection"] == "reflected"
        assert pair["reflection_max_abs_error"] == {
            "com_qpos": 0.0,
            "com_qvel": 0.0,
        }
        assert "qpos" not in original
        assert "qvel" not in original
        assert "absolute_altitude" not in original

    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"][TARGETS.name] == hashlib.sha256(TARGETS.read_bytes()).hexdigest()
