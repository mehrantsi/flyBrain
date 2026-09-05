from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, deque
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

MATERIALIZATION = "male-cns-v1.0-superclass-non-null-known-nt"
ANNOTATION_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)
NEUROTRANSMITTER_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
    "body-neurotransmitters-male-cns-v1.0.feather"
)
CONNECTIVITY_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)
DOWNLOAD_URL = "https://male-cns.janelia.org/download/"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
ARRAY_NAMES = ("destinations.npy", "neuron_ids.npy", "row_ptr.npy", "signed_counts.npy")
KNOWN_NT = {"acetylcholine", "gaba", "glutamate"}
NT_ALIASES = {"ach": "acetylcholine", "glu": "glutamate"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(ids: list[int]) -> str:
    payload = ",".join(str(value) for value in ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_feather(path: Path, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    try:
        from pyarrow import feather
    except ImportError as exc:  # pragma: no cover - dependency is declared by the package
        raise RuntimeError("pyarrow is required to read MaleCNS Feather files") from exc
    table = feather.read_table(path, columns=list(columns))
    return table.to_pylist()


def _canonical_nt(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().casefold()
    return NT_ALIASES.get(normalized, normalized)


def _require_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} is not a file: {path}")


def _load_pack(
    pack_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[int], dict[int, int]]:
    manifest_path = pack_path / "manifest.json"
    _require_file(manifest_path, "MaleCNS pack manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("materialization") != MATERIALIZATION:
        raise ValueError(
            f"MaleCNS artifact requires materialization {MATERIALIZATION}, "
            f"got {manifest.get('materialization')!r}"
        )
    expected_hashes = manifest.get("array_sha256")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(ARRAY_NAMES):
        raise ValueError("MaleCNS pack manifest must audit all four CSR arrays")
    arrays: dict[str, np.ndarray] = {}
    for name in ARRAY_NAMES:
        path = pack_path / name
        _require_file(path, f"MaleCNS pack array {name}")
        actual = _sha256(path)
        if actual != expected_hashes[name]:
            raise ValueError(f"MaleCNS pack array {name} hash does not match its manifest")
        arrays[name] = np.load(path, mmap_mode="r")
    ids = arrays["neuron_ids.npy"]
    row_ptr = arrays["row_ptr.npy"]
    destinations = arrays["destinations.npy"]
    signed_counts = arrays["signed_counts.npy"]
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("MaleCNS pack neuron_ids.npy must be a one-dimensional integer array")
    if ids.size == 0 or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("MaleCNS pack neuron IDs must be non-empty and strictly sorted")
    if row_ptr.ndim != 1 or row_ptr.size != ids.size + 1 or np.any(row_ptr[1:] < row_ptr[:-1]):
        raise ValueError("MaleCNS pack row_ptr.npy is inconsistent with neuron_ids.npy")
    if int(row_ptr[-1]) != destinations.size or destinations.size != signed_counts.size:
        raise ValueError("MaleCNS pack CSR arrays have inconsistent lengths")
    if destinations.size and np.any(destinations >= ids.size):
        raise ValueError("MaleCNS pack destinations are not valid CSR indices")
    if destinations.size:
        row_starts = row_ptr[:-1]
        row_ends = row_ptr[1:]
        for start, end in zip(row_starts, row_ends):
            start_i, end_i = int(start), int(end)
            if end_i - start_i > 1 and np.any(
                destinations[start_i + 1 : end_i] < destinations[start_i : end_i - 1]
            ):
                raise ValueError("MaleCNS pack destinations are not sorted within a CSR row")
    ids_list = [int(value) for value in ids]
    index_by_id = {root_id: index for index, root_id in enumerate(ids_list)}
    return manifest, arrays, ids_list, index_by_id


def _bind_source_hashes(
    manifest: dict[str, Any],
    sources: dict[str, Path],
) -> dict[str, str]:
    manifest_hashes = manifest.get("source_sha256", {})
    source_files = manifest.get("source_files", {})
    hashes: dict[str, str] = {}
    for role, path in sources.items():
        _require_file(path, role)
        expected = manifest_hashes.get(role)
        if expected is None and isinstance(source_files.get(role), dict):
            expected = source_files[role].get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"MaleCNS pack does not bind a SHA256 for {role}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"{role} source hash {actual} does not match the hash-bound MaleCNS pack {expected}"
            )
        hashes[role] = actual
    return hashes


def _side(row: dict[str, Any], field: str) -> str | None:
    value = row.get(field)
    return value if value in {"L", "R"} else None


def _group_resolution(ids: list[int], pack_ids: set[int]) -> dict[str, Any]:
    missing = sorted(root_id for root_id in ids if root_id not in pack_ids)
    return {
        "selected_count": len(ids),
        "present_count": len(ids) - len(missing),
        "missing_root_ids": missing,
    }


def _group(
    *,
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    selector: str,
    category: str,
    scope: str,
    side: str | None,
    pack_ids: set[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = [row for row in rows if predicate(row)]
    ids = sorted(int(row["bodyId"]) for row in selected)
    if not ids:
        raise ValueError(f"selection produced no rows: {selector}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"selection produced duplicate body IDs: {selector}")
    resolution = _group_resolution(ids, pack_ids)
    group: dict[str, Any] = {
        "evidence_category": category,
        "selector": selector,
        "biological_scope": scope,
        "root_ids": ids,
        "pack_resolution": resolution,
    }
    if side is not None:
        group["side"] = side
    return group, selected


def _type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(value): count
        for value, count in sorted(
            Counter(row.get("type") for row in rows).items(), key=lambda item: str(item[0])
        )
    }


def _nt_counts(rows: list[dict[str, Any]], nt_by_id: dict[int, str | None]) -> dict[str, int]:
    counts = Counter(nt_by_id.get(int(row["bodyId"])) or "<missing>" for row in rows)
    return {str(value): count for value, count in sorted(counts.items())}


def _representatives(
    ids: list[int],
    rows_by_id: dict[int, dict[str, Any]],
    side_field: str,
) -> list[int]:
    representatives: list[int] = []
    for side in ("L", "R"):
        candidates = [root_id for root_id in ids if _side(rows_by_id[root_id], side_field) == side]
        if candidates:
            representatives.append(min(candidates))
    if not representatives:
        representatives.append(min(ids))
    return representatives


def _shortest_route(
    source_id: int,
    target_ids: set[int],
    *,
    ids: list[int],
    index_by_id: dict[int, int],
    row_ptr: np.ndarray,
    destinations: np.ndarray,
    max_hops: int,
    max_visited: int = 100_000,
) -> list[int] | None:
    source = index_by_id[source_id]
    targets = {index_by_id[root_id] for root_id in target_ids}
    queue: deque[int] = deque([source])
    parent: dict[int, int | None] = {source: None}
    depth: dict[int, int] = {source: 0}
    while queue and len(parent) <= max_visited:
        current = queue.popleft()
        if current in targets:
            path: list[int] = []
            while current is not None:
                path.append(ids[current])
                current = parent[current]  # type: ignore[assignment]
            return path[::-1]
        if depth[current] >= max_hops:
            continue
        start = int(row_ptr[current])
        end = int(row_ptr[current + 1])
        for destination in destinations[start:end]:
            next_index = int(destination)
            if next_index in parent:
                continue
            parent[next_index] = current
            depth[next_index] = depth[current] + 1
            queue.append(next_index)
    return None


def _route_record(
    path: list[int],
    *,
    name: str,
    source_group: str,
    target_group: str,
    index_by_id: dict[int, int],
    row_ptr: np.ndarray,
    destinations: np.ndarray,
    signed_counts: np.ndarray,
    rows_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    edge_signed_counts: list[int] = []
    for pre_id, post_id in pairwise(path):
        pre_index = index_by_id[pre_id]
        post_index = index_by_id[post_id]
        start = int(row_ptr[pre_index])
        end = int(row_ptr[pre_index + 1])
        matches = np.flatnonzero(destinations[start:end] == post_index)
        if matches.size != 1:
            raise ValueError(f"route edge {pre_id}->{post_id} is not unique in packed CSR")
        edge_signed_counts.append(int(signed_counts[start + int(matches[0])]))
    node_annotations = []
    for root_id in path:
        row = rows_by_id[root_id]
        node_annotations.append(
            {
                "root_id": root_id,
                "superclass": row.get("superclass"),
                "class": row.get("class"),
                "subclass": row.get("subclass"),
                "type": row.get("type"),
                "side": _side(row, "rootSide") or _side(row, "somaSide"),
            }
        )
    return {
        "name": name,
        "source_group": source_group,
        "target_group": target_group,
        "source_root_id": path[0],
        "target_root_id": path[-1],
        "path_root_ids": path,
        "edge_signed_counts": edge_signed_counts,
        "node_annotations": node_annotations,
    }


def _route_evidence(
    groups: dict[str, dict[str, Any]],
    rows_by_id: dict[int, dict[str, Any]],
    arrays: dict[str, np.ndarray],
    ids: list[int],
    index_by_id: dict[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    row_ptr = arrays["row_ptr.npy"]
    destinations = arrays["destinations.npy"]
    signed_counts = arrays["signed_counts.npy"]
    queries = [
        ("taste_to_feeding_mn9", "taste_sugar", "feeding_mn9", 4),
        ("visual_loom_to_landing_left", "visual_loom_left", "landing_dn_left", 3),
        ("visual_loom_to_landing_right", "visual_loom_right", "landing_dn_right", 3),
        ("visual_motion_to_flight_start_left", "visual_motion_left", "flight_dng02_left", 4),
        ("visual_motion_to_flight_start_right", "visual_motion_right", "flight_dng02_right", 4),
        ("visual_motion_to_flight_altitude_left", "visual_motion_left", "flight_dng07_left", 4),
        ("visual_motion_to_flight_altitude_right", "visual_motion_right", "flight_dng07_right", 4),
        ("flight_feedback_to_flight_start_left", "flight_state_sapp_left", "flight_dng02_left", 4),
        (
            "flight_feedback_to_flight_start_right",
            "flight_state_sapp_right",
            "flight_dng02_right",
            4,
        ),
        ("flight_start_to_power_motor_left", "flight_dng02_left", "motor_flight_power_left", 3),
        ("flight_start_to_power_motor_right", "flight_dng02_right", "motor_flight_power_right", 3),
        (
            "flight_start_to_steering_motor_left",
            "flight_dng02_left",
            "motor_flight_steering_left",
            3,
        ),
        (
            "flight_start_to_steering_motor_right",
            "flight_dng02_right",
            "motor_flight_steering_right",
            3,
        ),
        ("walking_command_to_walking_motor_left", "walking_dn_left", "motor_walking_left", 3),
        ("walking_command_to_walking_motor_right", "walking_dn_right", "motor_walking_right", 3),
        ("landing_command_to_landing_motor_left", "landing_dn_left", "motor_landing_left", 3),
        ("landing_command_to_landing_motor_right", "landing_dn_right", "motor_landing_right", 3),
    ]
    routes: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for name, source_group, target_group, max_hops in queries:
        source_ids = groups[source_group]["root_ids"]
        target_ids = set(groups[target_group]["root_ids"])
        side_field = "rootSide" if source_group.startswith("flight_state_sapp") else "somaSide"
        sources = _representatives(source_ids, rows_by_id, side_field)
        found = None
        for source_id in sources:
            found = _shortest_route(
                source_id,
                target_ids,
                ids=ids,
                index_by_id=index_by_id,
                row_ptr=row_ptr,
                destinations=destinations,
                max_hops=max_hops,
            )
            if found is not None:
                break
        if found is None:
            unresolved.append(
                {
                    "name": name,
                    "source_group": source_group,
                    "target_group": target_group,
                    "source_root_ids_examined": sources,
                    "target_root_count": len(target_ids),
                    "max_hops": max_hops,
                    "reason": "no directed CSR route found within the bounded search depth",
                }
            )
            continue
        routes.append(
            _route_record(
                found,
                name=name,
                source_group=source_group,
                target_group=target_group,
                index_by_id=index_by_id,
                row_ptr=row_ptr,
                destinations=destinations,
                signed_counts=signed_counts,
                rows_by_id=rows_by_id,
            )
        )
    return routes, unresolved


def build_male_cns_io(
    annotations_path: str | Path,
    neurotransmitters_path: str | Path,
    connectivity_path: str | Path,
    pack_path: str | Path,
    output_path: str | Path,
    evidence_output_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    annotations = Path(annotations_path)
    neurotransmitters = Path(neurotransmitters_path)
    connectivity = Path(connectivity_path)
    pack = Path(pack_path)
    output = Path(output_path)
    evidence_output = Path(evidence_output_path)
    for path, name in (
        (annotations, "annotations"),
        (neurotransmitters, "neurotransmitters"),
        (connectivity, "connectivity"),
    ):
        _require_file(path, name)
    manifest, arrays, pack_ids_list, index_by_id = _load_pack(pack)
    source_hashes = _bind_source_hashes(
        manifest,
        {
            "annotations": annotations,
            "neurotransmitters": neurotransmitters,
            "connectivity": connectivity,
        },
    )

    annotation_rows = _read_feather(
        annotations,
        (
            "bodyId",
            "superclass",
            "class",
            "subclass",
            "entryNerve",
            "flywireType",
            "type",
            "rootSide",
            "somaSide",
            "instance",
            "status",
        ),
    )
    rows_by_id: dict[int, dict[str, Any]] = {}
    for row in annotation_rows:
        body_id = row.get("bodyId")
        if body_id is None:
            raise ValueError("MaleCNS annotations contain a null bodyId")
        body_id = int(body_id)
        if body_id in rows_by_id:
            raise ValueError(f"MaleCNS annotations contain duplicate bodyId {body_id}")
        row["bodyId"] = body_id
        rows_by_id[body_id] = row

    nt_rows = _read_feather(neurotransmitters, ("body", "consensus_nt"))
    nt_by_id: dict[int, str | None] = {}
    for row in nt_rows:
        body_id = row.get("body")
        if body_id is None:
            raise ValueError("MaleCNS neurotransmitter table contains a null body")
        body_id = int(body_id)
        if body_id in nt_by_id:
            raise ValueError(f"MaleCNS neurotransmitters contain duplicate body {body_id}")
        nt_by_id[body_id] = _canonical_nt(row.get("consensus_nt"))

    pack_ids = set(pack_ids_list)
    traced = lambda row: row.get("status") == "Traced"
    visual_types = {"H2", "HSE", "HSN", "HSS", "HST", "VS", "VST1", "VST2", "VSm"}
    wing_power_types = {"DLMn a, b", "DLMn c-f", "DVMn 1a-c", "DVMn 2a, b", "DVMn 3a, b"}
    wing_steering_types = {"b1 MN", "b2 MN", "b3 MN"}
    leg_walking_types = {"Ti extensor MN", "Tr extensor MN"}

    groups: dict[str, dict[str, Any]] = {}
    selected_rows: dict[str, list[dict[str, Any]]] = {}

    def add(
        name: str,
        predicate: Callable[[dict[str, Any]], bool],
        selector: str,
        category: str,
        scope: str,
        side: str | None,
    ) -> None:
        group, rows = _group(
            rows=annotation_rows,
            predicate=predicate,
            selector=selector,
            category=category,
            scope=scope,
            side=side,
            pack_ids=pack_ids,
        )
        groups[name] = group
        selected_rows[name] = rows

    for side, side_name in (("L", "left"), ("R", "right")):
        add(
            f"olfaction_{side_name}",
            lambda row, side=side: (
                traced(row)
                and row.get("superclass") == "cb_sensory"
                and row.get("class") == "olfactory"
                and row.get("entryNerve") == "AN"
                and str(row.get("type") or "").startswith("ORN_")
                and row.get("rootSide") == side
            ),
            "status == Traced and superclass == cb_sensory and class == olfactory and entryNerve == AN and type starts ORN_ and rootSide == {side}",
            "published_olfactory_sensory",
            "antenna olfactory receptor neurons with known left/right root side",
            side_name,
        )
        add(
            f"visual_motion_{side_name}",
            lambda row, side=side: (
                traced(row)
                and row.get("superclass") == "visual_projection"
                and row.get("type") in visual_types
                and row.get("somaSide") == side
                and _canonical_nt(nt_by_id.get(int(row["bodyId"]))) == "acetylcholine"
            ),
            "status == Traced and superclass == visual_projection and type in {H2,HSE,HSN,HSS,HST,VS,VST1,VST2,VSm} and somaSide == {side} and consensus_nt == acetylcholine",
            "published_visual_motion_input",
            "acetylcholine large-field optic-flow / motion projection neurons used as a visual input proxy",
            side_name,
        )
        add(
            f"visual_loom_{side_name}",
            lambda row, side=side: (
                traced(row)
                and row.get("superclass") == "visual_projection"
                and row.get("type") == "MeVP24"
                and row.get("somaSide") == side
                and _canonical_nt(nt_by_id.get(int(row["bodyId"]))) == "acetylcholine"
            ),
            "status == Traced and superclass == visual_projection and type == MeVP24 and somaSide == {side} and consensus_nt == acetylcholine",
            "published_visual_loom_input_proxy",
            "acetylcholine MeVP24 visual projection proxy for a loom/landing channel; no MaleCNS-specific loom annotation claim",
            side_name,
        )
        add(
            f"flight_state_sapp_{side_name}",
            lambda row, side=side: (
                traced(row)
                and row.get("superclass") == "sensory_ascending"
                and row.get("subclass") == "haltere"
                and row.get("type") == "SApp"
                and row.get("rootSide") == side
            ),
            "status == Traced and superclass == sensory_ascending and subclass == haltere and type == SApp and rootSide == {side}",
            "published_flight_ascending_feedback",
            "haltere ascending proprioceptive feedback candidate (SApp)",
            side_name,
        )
        add(
            f"flight_dng02_{side_name}",
            lambda row, side=side: (
                traced(row)
                and row.get("superclass") == "descending_neuron"
                and str(row.get("type") or "").startswith("DNg02")
                and row.get("somaSide") == side
            ),
            "status == Traced and superclass == descending_neuron and type starts DNg02 and somaSide == {side}",
            "candidate_flight_power_activation_descending",
            "DNg02 annotated flight-power activation candidate; not an isolated takeoff controller",
            side_name,
        )
        add(
            f"flight_dng07_{side_name}",
            lambda row, side=side: (
                traced(row)
                and row.get("superclass") == "descending_neuron"
                and row.get("type") == "DNg07"
                and row.get("somaSide") == side
            ),
            "status == Traced and superclass == descending_neuron and type == DNg07 and somaSide == {side}",
            "candidate_flight_power_decrease_descending",
            "DNg07 annotated power-decrease candidate; not an isolated altitude controller",
            side_name,
        )
        add(
            f"flight_steering_dn_{side_name}",
            lambda row, side=side: (
                traced(row)
                and row.get("superclass") == "descending_neuron"
                and row.get("type") in {"DNa03", "DNa04", "DNa15", "DNb01", "DNp03", "DNp28"}
                and row.get("somaSide") == side
            ),
            "status == Traced and superclass == descending_neuron and type in {DNa03,DNa04,DNa15,DNb01,DNp03,DNp28} and somaSide == {side}",
            "published_flight_steering_descending",
            "annotated flight steering descending neurons",
            side_name,
        )
        add(
            f"walking_dn_{side_name}",
            lambda row, side=side: (
                traced(row)
                and row.get("superclass") == "descending_neuron"
                and row.get("type") in {"DNa01", "DNa02", "DNa11", "DNg13"}
                and row.get("somaSide") == side
            ),
            "status == Traced and superclass == descending_neuron and type in {DNa01,DNa02,DNa11,DNg13} and somaSide == {side}",
            "published_walking_descending",
            "annotated walking descending command neurons",
            side_name,
        )
        add(
            f"landing_dn_{side_name}",
            lambda row, side=side: (
                traced(row)
                and row.get("superclass") == "descending_neuron"
                and row.get("type") in {"DNp07", "DNp10"}
                and row.get("somaSide") == side
            ),
            "status == Traced and superclass == descending_neuron and type in {DNp07,DNp10} and somaSide == {side}",
            "published_landing_descending",
            "annotated landing descending command neurons",
            side_name,
        )
        add(
            f"motor_flight_power_{side_name}",
            lambda row, side=side: (
                traced(row)
                and row.get("superclass") == "vnc_motor"
                and row.get("subclass") == "wm"
                and row.get("type") in wing_power_types
                and row.get("somaSide") == side
            ),
            "status == Traced and superclass == vnc_motor and subclass == wm and type in {DLMn a,b,DLMn c-f,DVMn 1a-c,DVMn 2a,b,DVMn 3a,b} and somaSide == {side}",
            "published_flight_wing_motor",
            "direct DLM/DVM wing motor neuron power pool",
            side_name,
        )
        add(
            f"motor_flight_steering_{side_name}",
            lambda row, side=side: (
                traced(row)
                and row.get("superclass") == "vnc_motor"
                and row.get("subclass") == "wm"
                and row.get("type") in wing_steering_types
                and row.get("somaSide") == side
            ),
            "status == Traced and superclass == vnc_motor and subclass == wm and type in {b1 MN,b2 MN,b3 MN} and somaSide == {side}",
            "published_flight_wing_motor",
            "direct b1/b2/b3 wing motor neuron steering pool",
            side_name,
        )
        add(
            f"motor_walking_{side_name}",
            lambda row, side=side: (
                traced(row)
                and row.get("superclass") == "vnc_motor"
                and row.get("subclass") in {"fl", "ml", "hl"}
                and row.get("type") in leg_walking_types
                and row.get("somaSide") == side
            ),
            "status == Traced and superclass == vnc_motor and subclass in {fl,ml,hl} and type in {Ti extensor MN,Tr extensor MN} and somaSide == {side}",
            "published_locomotor_leg_motor",
            "direct tibial/trochanter extensor leg motor pool for walking",
            side_name,
        )
        add(
            f"motor_landing_{side_name}",
            lambda row, side=side: (
                traced(row)
                and row.get("superclass") == "vnc_motor"
                and row.get("subclass") in {"fl", "ml", "hl"}
                and row.get("type") == "Ti extensor MN"
                and row.get("somaSide") == side
            ),
            "status == Traced and superclass == vnc_motor and subclass in {fl,ml,hl} and type == Ti extensor MN and somaSide == {side}",
            "published_locomotor_leg_motor",
            "direct tibial extensor leg motor pool used as landing readout",
            side_name,
        )

    taste_selector = (
        "status == Traced and superclass == cb_sensory and class == gustatory and "
        "subclass == labellar bristle and entryNerve == MxLbN and flywireType == LB3"
    )
    add(
        "taste_sugar",
        lambda row: (
            traced(row)
            and row.get("superclass") == "cb_sensory"
            and row.get("class") == "gustatory"
            and row.get("subclass") == "labellar bristle"
            and row.get("entryNerve") == "MxLbN"
            and row.get("flywireType") == "LB3"
        ),
        taste_selector,
        "published_gustatory_sensory",
        "labellar LB3 gustatory receptor neurons; source annotation classifies this cell type as sugar/water related",
        "both",
    )
    add(
        "feeding_mn9",
        lambda row: (
            traced(row)
            and row.get("superclass") == "cb_motor"
            and row.get("type") == "MN9"
            and row.get("somaSide") == "L"
            and row.get("instance") == "MN9_L"
        ),
        "status == Traced and superclass == cb_motor and type == MN9 and somaSide == L and instance == MN9_L",
        "published_feeding_motor",
        "left MaleCNS MN9 feeding motor counterpart",
        "left",
    )
    if len(selected_rows["feeding_mn9"]) != 1:
        raise ValueError("MaleCNS feeding_mn9 selection must identify exactly one MN9_L")

    food_channels: dict[str, dict[str, Any]] = {}
    for glomerulus, band in (
        ("DM1", "attractive"),
        ("DM2", "core"),
        ("DM3", "high_concentration"),
    ):
        for side, side_name in (("L", "left"), ("R", "right")):
            channel_rows = [
                row
                for row in selected_rows[f"olfaction_{side_name}"]
                if row.get("type") == f"ORN_{glomerulus}"
            ]
            channel_ids = sorted(int(row["bodyId"]) for row in channel_rows)
            if not channel_ids:
                raise ValueError(f"food channel {glomerulus}_{side_name} is empty")
            food_channels[f"{glomerulus.casefold()}_{side_name}"] = {
                "glomerulus": glomerulus,
                "side": side_name,
                "response_band": band,
                "root_ids": channel_ids,
            }

    group_pack_resolution = {
        name: value["pack_resolution"] for name, value in sorted(groups.items())
    }
    summary_categories: dict[str, dict[str, int]] = {}
    for group in groups.values():
        category = group["evidence_category"]
        resolution = group["pack_resolution"]
        counts = summary_categories.setdefault(
            category,
            {"groups": 0, "selected_root_ids": 0, "present_root_ids": 0, "missing_root_ids": 0},
        )
        counts["groups"] += 1
        counts["selected_root_ids"] += resolution["selected_count"]
        counts["present_root_ids"] += resolution["present_count"]
        counts["missing_root_ids"] += len(resolution["missing_root_ids"])
    summary = {
        "group_count": len(groups),
        "selected_root_ids": sum(
            value["pack_resolution"]["selected_count"] for value in groups.values()
        ),
        "present_root_ids": sum(
            value["pack_resolution"]["present_count"] for value in groups.values()
        ),
        "missing_root_ids": sum(
            len(value["pack_resolution"]["missing_root_ids"]) for value in groups.values()
        ),
        "category_counts": summary_categories,
    }
    array_hashes = {name: manifest["array_sha256"][name] for name in sorted(ARRAY_NAMES)}
    dataset = {
        "name": "MaleCNS v1.0 annotated sensorimotor I/O",
        "materialization": MATERIALIZATION,
        "annotation_release": "MaleCNS v1.0",
        "annotation_commit": source_hashes["annotations"],
        "annotation_source": ANNOTATION_URL,
        "annotation_sha256": source_hashes["annotations"],
        "pack_neuron_ids_sha256": array_hashes["neuron_ids.npy"],
        "pack_array_sha256": array_hashes,
        "pack_path": "outputs/packs/male_cns_v1",
        "selection_provenance": (
            "Selections are generated from exact MaleCNS v1.0 annotation fields and the published "
            "consensus_nt table; every source Feather hash is checked against the selected CSR pack. "
            "Unknown-side ORNs and unsupported histamine photoreceptors are retained as explicit "
            "exclusions in the supplemental evidence rather than assigned to a side or channel."
        ),
    }
    artifact = {
        "schema_version": 1,
        "dataset": dataset,
        "groups": groups,
        "summary": summary,
        "pack_resolution": {
            "materialization": MATERIALIZATION,
            "pack_neuron_count": len(pack_ids_list),
            "pack_neuron_ids_sha256": array_hashes["neuron_ids.npy"],
            "groups": group_pack_resolution,
        },
        "food_olfaction": {
            "schema": "flybrain-food-olfaction-v1",
            "reference_odor": "apple_cider_vinegar",
            "concentration_unit": "isobutylene-equivalent ppm",
            "evidence_source": "https://doi.org/10.1038/nature07983",
            "annotation_field": "type",
            "selection_rule": (
                "status == Traced and superclass == cb_sensory and class == olfactory and "
                "entryNerve == AN and type == ORN_{glomerulus} and rootSide == {side}"
            ),
            "channels": food_channels,
        },
    }

    routes, unresolved = _route_evidence(
        artifact["groups"], rows_by_id, arrays, pack_ids_list, index_by_id
    )
    evidence_groups: dict[str, dict[str, Any]] = {}
    for name in sorted(groups):
        rows = selected_rows[name]
        evidence_groups[name] = {
            "selector": groups[name]["selector"],
            "selected_count": len(rows),
            "present_in_pack_count": groups[name]["pack_resolution"]["present_count"],
            "root_ids_sha256": _ids_sha256(groups[name]["root_ids"]),
            "type_counts": _type_counts(rows),
            "consensus_nt_counts": _nt_counts(rows, nt_by_id),
        }
    photoreceptor_rows = [
        row
        for row in annotation_rows
        if traced(row) and row.get("superclass") == "ol_sensory" and row.get("type") == "R1-R6"
    ]
    unknown_orn_rows = [
        row
        for row in annotation_rows
        if traced(row)
        and row.get("superclass") == "cb_sensory"
        and row.get("class") == "olfactory"
        and row.get("entryNerve") == "AN"
        and str(row.get("type") or "").startswith("ORN_")
        and row.get("rootSide") not in {"L", "R"}
    ]
    evidence = {
        "schema_version": 1,
        "dataset": dataset,
        "evidence_references": [
            {
                "claim": "FlyWire/VFB classifies LB3 as an adult gustatory sugar/water receptor-neuron type.",
                "source": "https://www.virtualflybrain.org/term/gng.2078-flywire720575940640649691-vfb_fw045435/",
            },
            {
                "claim": "The apple-cider-vinegar food-odor profile is anchored to the published olfactory response study.",
                "source": "https://doi.org/10.1038/nature07983",
            },
            {
                "claim": "MaleCNS v1.0 download and licensing provenance.",
                "source": DOWNLOAD_URL,
            },
        ],
        "source_files": {
            "annotations": {
                "path": str(annotations),
                "sha256": source_hashes["annotations"],
                "url": ANNOTATION_URL,
            },
            "neurotransmitters": {
                "path": str(neurotransmitters),
                "sha256": source_hashes["neurotransmitters"],
                "url": NEUROTRANSMITTER_URL,
            },
            "connectivity": {
                "path": str(connectivity),
                "sha256": source_hashes["connectivity"],
                "url": CONNECTIVITY_URL,
            },
            "pack_manifest": {
                "path": str(pack / "manifest.json"),
                "sha256": _sha256(pack / "manifest.json"),
                "url": DOWNLOAD_URL,
            },
        },
        "group_census": evidence_groups,
        "excluded_inputs": [
            {
                "name": "olfaction_unknown_root_side",
                "selector": "status == Traced and superclass == cb_sensory and class == olfactory and entryNerve == AN and type starts ORN_ and rootSide not in {L,R}",
                "count": len(unknown_orn_rows),
                "reason": "No side assignment is present; these receptors are not silently grafted into left or right channels.",
            },
            {
                "name": "compound_eye_R1-R6",
                "selector": "status == Traced and superclass == ol_sensory and type == R1-R6",
                "count": len(photoreceptor_rows),
                "consensus_nt_counts": _nt_counts(photoreceptor_rows, nt_by_id),
                "reason": "Published consensus is histamine; MaleCNS CSR omits unknown/histamine outgoing edges, so no dead retinal input channel is exposed.",
            },
            {
                "name": "feeding_mn9_right",
                "selector": "status == Traced and superclass == cb_motor and type == MN9 and somaSide == R and instance == MN9_R",
                "count": 1,
                "root_ids": [16949],
                "reason": "The bridge's single feeding readout is bound to the left MN9 counterpart (10331); the right counterpart remains documented here rather than duplicated as a probe.",
            },
        ],
        "routes": routes,
        "unresolved_route_queries": unresolved,
        "route_policy": {
            "direction": "packed CSR presynaptic body -> postsynaptic body",
            "maximum_hops": 4,
            "selection": "one deterministic shortest directed route per named query; absent routes remain explicit unresolved queries",
            "edge_weights": "edge_signed_counts are copied from signed_counts.npy; no inferred or grafted edges",
        },
        "unknowns": [
            "SApp is an annotated haltere ascending feedback candidate; the bridge exposes it through an engineered scalar angular-speed proxy, not a measured full 3D haltere signal.",
            "DNg02 is an annotated flight-power activation population and is not asserted to be a specific takeoff controller.",
            "DNg07 is an unisolated flight-power-decrease candidate, not an isolated altitude controller; coactivation with other descending types remains possible.",
            "DLM/DVM and extensor motor groups include published consensus_nt == unclear rows; their incoming synapses remain in the pack while unknown-NT outgoing edges are omitted by pack policy.",
            "Visual motion and MeVP24 groups are acetylcholine projection proxies; MeVP24 is not claimed to be a verified loom neuron, and no R1-R6 histamine input path is exposed.",
            "No muscle-to-body torque mapping is asserted by this artifact; motor groups are neural population readouts only.",
        ],
    }
    _write_json(output, artifact)
    _write_json(evidence_output, evidence)
    return artifact, evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a hash-bound MaleCNS neural I/O artifact")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--neurotransmitters", type=Path, required=True)
    parser.add_argument("--connectivity", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact, evidence = build_male_cns_io(
        annotations_path=args.annotations,
        neurotransmitters_path=args.neurotransmitters,
        connectivity_path=args.connectivity,
        pack_path=args.pack,
        output_path=args.output,
        evidence_output_path=args.evidence_output,
    )
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "evidence": str(args.evidence_output),
                "groups": artifact["summary"],
                "routes": {
                    "resolved": len(evidence["routes"]),
                    "unresolved": len(evidence["unresolved_route_queries"]),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
