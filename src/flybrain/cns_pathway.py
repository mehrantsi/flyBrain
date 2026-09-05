from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

from pyarrow import feather

from .connectome import PackedConnectome
from .pack import _sha256_file

EVIDENCE = [
    "https://male-cns.janelia.org/download/",
    "https://www.nature.com/articles/s41593-019-0413-4",
    "https://www.nature.com/articles/s41586-026-10735-w",
    "https://www.virtualflybrain.org/blog/2022/01/01/mevp24-fbbt_20007653/",
]


def motor_paths(pack, annotations, inputs, relays, motors):
    indices = {int(body): i for i, body in enumerate(pack.neuron_ids)}
    missing = (set(inputs) | set(relays) | set(motors)) - indices.keys()
    if missing:
        raise ValueError(f"Selected neurons missing from pack: {sorted(missing)}")

    def successors(body):
        i = indices[body]
        edges = range(int(pack.row_ptr[i]), int(pack.row_ptr[i + 1]))
        return sorted(int(pack.neuron_ids[pack.destinations[e]]) for e in edges)

    queue = deque()
    visited = set()
    for source in sorted(inputs):
        for relay in successors(source):
            if relay in relays:
                queue.append([source, relay])
                visited.add(relay)
    if visited != set(relays):
        raise ValueError("Every relay must have a measured direct connection from an input")
    found = {}
    while queue:
        path = queue.popleft()
        for target in successors(path[-1]):
            if target in path:
                continue
            if target in motors:
                found.setdefault(target, path + [target])
            elif (
                len(path) < 4
                and target not in visited
                and annotations.get(target, {}).get("superclass") == "vnc_intrinsic"
            ):
                visited.add(target)
                queue.append(path + [target])
    missing = set(motors) - found.keys()
    if missing:
        raise ValueError(f"No measured path through at most two VNC interneurons: {sorted(missing)}")
    return [found[body] for body in sorted(motors)]


def build_pathway(pack, annotations):
    def typed(kind, superclass):
        return sorted(
            body for body, row in annotations.items()
            if row["type"] == kind and row["superclass"] == superclass
        )

    inputs = typed("MeVP24", "visual_projection")
    relays = typed("DNp10", "descending_neuron")
    motors = typed("Ti extensor MN", "vnc_motor")
    if (len(inputs), len(relays), len(motors)) != (2, 2, 12):
        raise ValueError("Expected MaleCNS v1.0 census: 2 MeVP24, 2 DNp10, 12 tibial extensors")
    groups = {}
    for body in motors:
        row = annotations[body]
        leg, side = row["subclass"], row["somaSide"]
        if leg not in ("fl", "ml", "hl") or side not in ("L", "R"):
            raise ValueError(f"Unresolved leg/side for motor neuron {body}")
        groups.setdefault(f"tibia_extensor_{leg}_{side}", []).append(body)
    if len(groups) != 6 or any(len(ids) != 2 for ids in groups.values()):
        raise ValueError("Expected two annotated tibial extensor neurons per leg")
    paths = []
    for source in inputs:
        side = annotations[source]["somaSide"]
        same_side_relays = [relay for relay in relays if annotations[relay]["somaSide"] == side]
        if side not in ("L", "R") or len(same_side_relays) != 1:
            raise ValueError("Expected one ipsilateral DNp10 per MeVP24")
        paths.extend(motor_paths(pack, annotations, [source], same_side_relays, motors))
    return {
        "schema_version": 1,
        "name": "MaleCNS MeVP24-DNp10 tibial-extensor pathway",
        "materialization": pack.manifest["materialization"],
        "pack_array_sha256": pack.manifest["array_sha256"],
        "evidence": EVIDENCE,
        "interpretation": (
            "Experimental visual-projection activation, not retinal looming transduction. "
            "MeVP24 is selected by annotation and measured connectivity, not validated looming "
            "tuning. DNp10 has published landing leg-extension evidence. Shortest anatomical "
            "paths are witnesses only; the entire imported CNS graph is simulated. "
            "Shiu LIF parameters and transmitter signs are unvalidated for this CNS. "
            "Motor spikes are not muscle activation or a validated landing maneuver. "
            "No prediction of flexor inhibition is tested against a silent baseline."
        ),
        "stimulus_ids": inputs,
        "relay_ids": relays,
        "readout_groups": dict(sorted(groups.items())),
        "anatomical_paths": paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Compile the MaleCNS landing-pathway assay")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pack = PackedConnectome.load(args.pack)
    for name, expected in pack.manifest["array_sha256"].items():
        if _sha256_file(args.pack / name) != expected:
            raise ValueError(f"Pack hash mismatch: {name}")
    annotation_hash = _sha256_file(args.annotations)
    if annotation_hash not in pack.manifest["source_sha256"].values():
        raise ValueError("Annotations do not match an audited pack source")
    rows = feather.read_table(args.annotations).to_pylist()
    annotations = {row["bodyId"]: row for row in rows}
    if len(annotations) != len(rows):
        raise ValueError("Duplicate body IDs in annotation table")
    pathway = build_pathway(pack, annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(pathway, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"pathway": str(args.output), "motor_neurons": 12}))


if __name__ == "__main__":
    main()
