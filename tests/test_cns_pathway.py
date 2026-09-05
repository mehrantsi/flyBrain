import pytest

from flybrain.cns_pathway import build_pathway, motor_paths
from flybrain.connectome import PackedConnectome
from tools.verify_male_cns_pathway import summarize


def test_paths_follow_csr_and_only_traverse_vnc_interneurons():
    pack = PackedConnectome.from_arrays(
        [10, 20, 30, 40, 50], [0, 1, 3, 4, 4, 4], [1, 2, 3, 4], [93, -5, 30, 4]
    )
    annotations = {30: {"superclass": "vnc_intrinsic"}}
    assert motor_paths(pack, annotations, [10], [20], [40, 50]) == [
        [10, 20, 40], [10, 20, 30, 50]
    ]
    with pytest.raises(ValueError, match="No measured path"):
        motor_paths(pack, {}, [10], [20], [50])


def test_paths_reject_missing_cells_and_disconnected_relays():
    pack = PackedConnectome.from_arrays([10, 20, 30], [0, 1, 1, 1], [2], [93])
    with pytest.raises(ValueError, match="missing from pack"):
        motor_paths(pack, {}, [10], [20], [40])
    with pytest.raises(ValueError, match="Every relay"):
        motor_paths(pack, {}, [10], [20], [30])


def test_builder_does_not_silently_accept_partial_cell_type_census():
    pack = PackedConnectome.from_arrays([10], [0, 0], [], [])
    with pytest.raises(ValueError, match="census"):
        build_pathway(pack, {})


def test_builder_binds_all_six_motor_pools_to_pack_identity():
    annotations = {
        10: {"type": "MeVP24", "superclass": "visual_projection"},
        11: {"type": "MeVP24", "superclass": "visual_projection"},
        20: {"type": "DNp10", "superclass": "descending_neuron"},
        21: {"type": "DNp10", "superclass": "descending_neuron"},
    }
    for body, annotation in annotations.items():
        annotation["somaSide"] = "L" if body % 2 == 0 else "R"
    for i, (leg, side) in enumerate((leg, side) for leg in ("fl", "ml", "hl") for side in "LR"):
        for offset in range(2):
            annotations[30 + 2 * i + offset] = {
                "type": "Ti extensor MN", "superclass": "vnc_motor",
                "subclass": leg, "somaSide": side,
            }
    pack = PackedConnectome.from_arrays(
        sorted(annotations), [0, 1, 2, 14, 26] + [26] * 12,
        [2, 3] + list(range(4, 16)) * 2, [10] * 26,
        {"materialization": "test-cns", "array_sha256": {"fixture": "test"}},
    )
    pathway = build_pathway(pack, annotations)
    assert pathway["pack_array_sha256"] == pack.manifest["array_sha256"]
    assert len(pathway["readout_groups"]) == 6
    assert len(pathway["anatomical_paths"]) == 24
    assert all(len(path) == 3 for path in pathway["anatomical_paths"])


def test_summary_keeps_software_validity_separate_from_pathway_success():
    def report(inputs, relays, motors, events):
        return {
            "inputs": {"total_spikes": inputs},
            "relays": {"total_spikes": relays, "active_neurons": int(relays > 0)},
            "readouts": {"motor": {"total_spikes": motors, "active_neurons": int(motors > 0)}},
            "population": {"total_spikes": inputs + relays + motors},
            "stimulus": {"event_count": events},
        }
    runs = {
        "intact": report(10, 3, 4, 8),
        "no-input": report(0, 0, 0, 0),
        "input-disconnected": report(8, 0, 0, 8),
        "relay-disconnected": report(10, 3, 1, 8),
    }
    summary = summarize({1: runs})
    assert summary["software_controls_pass"]
    assert summary["pathway_response_reproduced_all_seeds"]
    assert summary["checks"][0]["relay_disconnect_motor_reduction_fraction"] == 0.75
    runs["relay-disconnected"] = report(10, 3, 5, 8)
    summary = summarize({1: runs})
    assert summary["software_controls_pass"]
    assert not summary["pathway_response_reproduced_all_seeds"]
    assert not summary["live_default_changed"]
