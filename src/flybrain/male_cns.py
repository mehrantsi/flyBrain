from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

_ANNOTATION_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)
_NEUROTRANSMITTER_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
    "body-neurotransmitters-male-cns-v1.0.feather"
)
_CONNECTIVITY_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)
_DOWNLOAD_PAGE_URL = "https://male-cns.janelia.org/download/"
_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
_LICENSE_NOTICE = "The Male CNS dataset is licensed under CC-BY."

_ANNOTATION_COLUMNS = ("bodyId", "superclass", "status")
_NEUROTRANSMITTER_COLUMNS = ("body", "consensus_nt")
_CONNECTIVITY_COLUMNS = ("body_pre", "body_post", "weight")
_KNOWN_SIGNS = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}
_NT_ALIASES = {"ach": "acetylcholine", "glu": "glutamate"}
_EDGE_DTYPE = np.dtype(
    [("pre", "<u4"), ("post", "<u4"), ("weight", "<u2"), ("sign", "i1")]
)
_CHUNK_ROWS = 1_000_000
_UINT32_MAX = int(np.iinfo(np.uint32).max)
_UINT64_MAX = int(np.iinfo(np.uint64).max)
_INT64_MAX = int(np.iinfo(np.int64).max)
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT16_MAX = int(np.iinfo(np.int16).max)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_arrow() -> tuple[Any, Any]:
    try:
        from pyarrow import feather, ipc
    except ImportError as exc:  # pragma: no cover - dependency is declared by the package
        raise RuntimeError("pyarrow is required to import MaleCNS Feather files") from exc
    return feather, ipc


def _schema(path: Path, ipc: Any) -> Any:
    return ipc.open_file(path).schema


def _schema_description(schema: Any) -> list[dict[str, str]]:
    return [{"name": field.name, "type": str(field.type)} for field in schema]


def _require_columns(schema: Any, required: tuple[str, ...], source_name: str) -> None:
    missing = [name for name in required if name not in schema.names]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {', '.join(missing)}")


def _integer_array(column: Any, *, name: str, nonnegative: bool) -> np.ndarray:
    if column.null_count:
        raise ValueError(f"{name} contains null values")
    values = np.asarray(column.to_numpy(zero_copy_only=False))
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer column")
    if values.size and values.dtype.kind == "i":
        if nonnegative and np.any(values < 0):
            raise ValueError(f"{name} contains a negative value")
        if int(values.min()) < _INT64_MIN or int(values.max()) > _INT64_MAX:
            raise ValueError(f"{name} is outside signed int64 range")
    if values.size and values.dtype.kind == "u" and int(values.max()) > _UINT64_MAX:
        raise ValueError(f"{name} is outside uint64 range")
    if nonnegative:
        return values.astype(np.uint64, copy=False)
    if values.dtype.kind == "u" and values.size and int(values.max()) > _INT64_MAX:
        raise ValueError(f"{name} is outside signed int64 range")
    return values.astype(np.int64, copy=False)


def _text_values(column: Any, *, name: str, allow_null: bool) -> list[str | None]:
    values = column.to_pylist()
    for row, value in enumerate(values):
        if value is None:
            if allow_null:
                continue
            raise ValueError(f"{name} contains a null value at row {row}")
        if not isinstance(value, str):
            raise ValueError(f"{name} must contain strings or null values")  # noqa: TRY004
    return values


def _count_records(values: list[str | None]) -> list[dict[str, Any]]:
    counts = Counter(values)
    return _counter_records(counts)


def _counter_records(counts: Counter[str | None]) -> list[dict[str, Any]]:
    ordered = sorted(counts.items(), key=lambda item: (item[0] is not None, str(item[0])))
    return [{"value": value, "count": int(count)} for value, count in ordered]


def _nt_sign(label: str) -> int:
    normalized = label.strip().casefold()
    canonical = _NT_ALIASES.get(normalized, normalized)
    return _KNOWN_SIGNS.get(canonical, 0)


def _membership(ids: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not ids.size:
        return np.zeros(values.size, dtype=bool), np.zeros(values.size, dtype=np.int64)
    indices = np.searchsorted(ids, values)
    safe = np.minimum(indices, ids.size - 1)
    present = (indices < ids.size) & (ids[safe] == values)
    return present, indices


def _flush_chunk(
    pending: list[np.ndarray],
    *,
    chunk_dir: Path,
    chunk_index: int,
) -> tuple[list[np.ndarray], int, Path]:
    records = np.concatenate(pending) if len(pending) > 1 else pending[0]
    order = np.lexsort((records["post"], records["pre"]))
    records = records[order]
    path = chunk_dir / f"chunk-{chunk_index:06d}.bin"
    records.tofile(path)
    return [], 0, path


def _materialize_chunks(
    paths: list[Path],
    *,
    expected_materialized_edges: int,
    destinations: np.ndarray,
    signed_counts: np.ndarray,
) -> int:
    if paths:
        chunks = [np.memmap(path, dtype=_EDGE_DTYPE, mode="r") for path in paths]
        records = np.concatenate(chunks)
        del chunks
    else:
        records = np.empty(0, dtype=_EDGE_DTYPE)
    order = np.lexsort((records["post"], records["pre"]))
    records = records[order]
    del order
    if records.size > 1:
        duplicate = (records["pre"][1:] == records["pre"][:-1]) & (
            records["post"][1:] == records["post"][:-1]
        )
        if np.any(duplicate):
            position = int(np.flatnonzero(duplicate)[0] + 1)
            raise ValueError(
                "duplicate presynaptic/postsynaptic pair "
                f"at selected indices ({int(records['pre'][position])}, "
                f"{int(records['post'][position])})"
            )
    known = records["sign"] != 0
    if int(np.count_nonzero(known)) != expected_materialized_edges:
        raise ValueError(
            "materialized edge count changed between passes: "
            f"expected {expected_materialized_edges}, wrote {int(np.count_nonzero(known))}"
        )
    if expected_materialized_edges:
        destinations[:] = records["post"][known]
        signed_counts[:] = (
            records["sign"][known].astype(np.int16)
            * records["weight"][known].astype(np.int16)
        )
    return expected_materialized_edges


def _write_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def import_male_cns(
    annotations_path: str | Path,
    neurotransmitters_path: str | Path,
    connectivity_path: str | Path,
    output_path: str | Path,
    *,
    materialization: str = "male-cns-v1.0-superclass-non-null-known-nt",
) -> dict[str, Any]:
    """Compile the published MaleCNS v1.0 tables into the project's CSR pack format.

    Nodes are the annotation rows whose ``superclass`` is non-null.  Connectivity is
    retained only when both endpoints are selected and the presynaptic body's published
    ``consensus_nt`` is acetylcholine, GABA, or glutamate.  The corresponding signs are
    +1, -1, and -1.  Unknown, modulatory, and missing NT labels retain their nodes but
    have their outgoing edges omitted; the omission is recorded in the manifest.
    """

    annotation = Path(annotations_path)
    neurotransmitters = Path(neurotransmitters_path)
    connectivity = Path(connectivity_path)
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output directory already exists: {output}")
    for path, name in (
        (annotation, "annotation"),
        (neurotransmitters, "neurotransmitter"),
        (connectivity, "connectivity"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} source file does not exist: {path}")

    feather, ipc = _load_arrow()
    annotation_schema = _schema(annotation, ipc)
    nt_schema = _schema(neurotransmitters, ipc)
    connectivity_schema = _schema(connectivity, ipc)
    _require_columns(annotation_schema, _ANNOTATION_COLUMNS, "annotation Feather")
    _require_columns(nt_schema, _NEUROTRANSMITTER_COLUMNS, "neurotransmitter Feather")
    _require_columns(connectivity_schema, _CONNECTIVITY_COLUMNS, "connectivity Feather")

    annotation_table = feather.read_table(annotation, columns=list(_ANNOTATION_COLUMNS))
    annotation_ids = _integer_array(
        annotation_table["bodyId"], name="annotation bodyId", nonnegative=True
    )
    if np.unique(annotation_ids).size != annotation_ids.size:
        raise ValueError("annotation bodyId values must be unique")
    annotation_superclasses = _text_values(
        annotation_table["superclass"], name="annotation superclass", allow_null=True
    )
    annotation_status = _text_values(
        annotation_table["status"], name="annotation status", allow_null=True
    )
    selected_mask = np.fromiter(
        (value is not None for value in annotation_superclasses), dtype=bool, count=annotation_ids.size
    )
    selected_ids = np.sort(annotation_ids[selected_mask])
    if selected_ids.size > _UINT32_MAX:
        raise ValueError("selected neuron count exceeds uint32 limit")

    nt_table = feather.read_table(neurotransmitters, columns=list(_NEUROTRANSMITTER_COLUMNS))
    nt_ids_unsorted = _integer_array(nt_table["body"], name="neurotransmitter body", nonnegative=True)
    if np.unique(nt_ids_unsorted).size != nt_ids_unsorted.size:
        raise ValueError("neurotransmitter body values must be unique")
    nt_labels_unsorted = _text_values(
        nt_table["consensus_nt"], name="neurotransmitter consensus_nt", allow_null=False
    )
    nt_order = np.argsort(nt_ids_unsorted)
    nt_ids = nt_ids_unsorted[nt_order]
    nt_labels = np.asarray(nt_labels_unsorted, dtype=object)[nt_order]
    nt_signs = np.fromiter(
        (_nt_sign(str(value)) for value in nt_labels),
        dtype=np.int8,
        count=nt_labels.size,
    )

    selected_nt_positions = np.searchsorted(nt_ids, selected_ids)
    selected_nt_safe = np.minimum(selected_nt_positions, max(nt_ids.size - 1, 0))
    selected_nt_present = (selected_nt_positions < nt_ids.size) & (
        nt_ids[selected_nt_safe] == selected_ids
    ) if nt_ids.size else np.zeros(selected_ids.size, dtype=bool)
    selected_node_labels: list[str | None] = []
    for position, present in zip(selected_nt_positions, selected_nt_present, strict=True):
        selected_node_labels.append(str(nt_labels[position]) if present else "<missing>")

    source_hashes = {
        "annotations": _sha256(annotation),
        "neurotransmitters": _sha256(neurotransmitters),
        "connectivity": _sha256(connectivity),
    }
    source_urls = {
        "annotations": _ANNOTATION_URL,
        "neurotransmitters": _NEUROTRANSMITTER_URL,
        "connectivity": _CONNECTIVITY_URL,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output.name}.male-cns-", dir=str(output.parent)))
    chunk_dir = temporary_root / "chunks"
    staging = temporary_root / "staging"
    chunk_dir.mkdir()
    staging.mkdir()
    try:
        pending: list[np.ndarray] = []
        pending_rows = 0
        chunk_paths: list[Path] = []
        chunk_index = 0
        row_counts = np.zeros(selected_ids.size, dtype=np.uint64)
        raw_rows = 0
        raw_contact_sum = 0
        raw_weight_min: int | None = None
        raw_weight_max: int | None = None
        raw_self_loops = 0
        selected_pre_rows = 0
        selected_post_rows = 0
        selected_rows = 0
        selected_contact_sum = 0
        materialized_edges = 0
        materialized_contact_sum = 0
        excitatory_edges = 0
        inhibitory_edges = 0
        unknown_edges = 0
        unknown_contact_sum = 0
        unknown_edge_labels: Counter[str] = Counter()

        reader = ipc.open_file(connectivity)
        if reader.schema.names != connectivity_schema.names:
            raise ValueError("connectivity Feather schema changed while opening")
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            pre = _integer_array(
                batch.column(batch.schema.get_field_index("body_pre")),
                name="connectivity body_pre",
                nonnegative=True,
            )
            post = _integer_array(
                batch.column(batch.schema.get_field_index("body_post")),
                name="connectivity body_post",
                nonnegative=True,
            )
            weights = _integer_array(
                batch.column(batch.schema.get_field_index("weight")),
                name="connectivity weight",
                nonnegative=False,
            )
            if pre.size != post.size or pre.size != weights.size:
                raise ValueError(f"connectivity columns have inconsistent lengths in batch {batch_index}")
            if weights.size:
                if np.any(weights <= 0):
                    raise ValueError(f"connectivity weight must be positive in batch {batch_index}")
                if np.any(weights > _INT16_MAX):
                    raise ValueError(
                        f"connectivity weight exceeds signed int16 range in batch {batch_index}"
                    )
                batch_sum = int(weights.astype(np.uint64).sum(dtype=np.uint64))
                raw_contact_sum += batch_sum
                if raw_contact_sum > _UINT64_MAX:
                    raise ValueError("raw connectivity contact sum exceeds uint64 range")
                batch_min = int(weights.min())
                batch_max = int(weights.max())
                raw_weight_min = batch_min if raw_weight_min is None else min(raw_weight_min, batch_min)
                raw_weight_max = batch_max if raw_weight_max is None else max(raw_weight_max, batch_max)
                raw_self_loops += int(np.count_nonzero(pre == post))
            raw_rows += int(pre.size)

            pre_present, pre_indices = _membership(selected_ids, pre)
            post_present, post_indices = _membership(selected_ids, post)
            selected_pre_rows += int(np.count_nonzero(pre_present))
            selected_post_rows += int(np.count_nonzero(post_present))
            both = pre_present & post_present
            if not np.any(both):
                continue

            selected_pre = pre[both]
            selected_pre_indices = pre_indices[both]
            selected_post_indices = post_indices[both]
            selected_weights = weights[both]
            nt_positions = np.searchsorted(nt_ids, selected_pre)
            nt_safe = np.minimum(nt_positions, max(nt_ids.size - 1, 0))
            nt_present = (nt_positions < nt_ids.size) & (nt_ids[nt_safe] == selected_pre) if nt_ids.size else np.zeros(selected_pre.size, dtype=bool)
            signs = np.zeros(selected_pre.size, dtype=np.int8)
            signs[nt_present] = nt_signs[nt_positions[nt_present]]
            selected_labels = np.full(selected_pre.size, "<missing>", dtype=object)
            selected_labels[nt_present] = nt_labels[nt_positions[nt_present]]

            selected_count = int(selected_pre.size)
            selected_rows += selected_count
            batch_selected_sum = int(selected_weights.astype(np.uint64).sum(dtype=np.uint64))
            selected_contact_sum += batch_selected_sum
            if selected_contact_sum > _UINT64_MAX:
                raise ValueError("selected connectivity contact sum exceeds uint64 range")
            unknown = signs == 0
            known = ~unknown
            unknown_count = int(np.count_nonzero(unknown))
            unknown_edges += unknown_count
            unknown_sum = int(selected_weights[unknown].astype(np.uint64).sum(dtype=np.uint64))
            unknown_contact_sum += unknown_sum
            if unknown_contact_sum > _UINT64_MAX:
                raise ValueError("omitted unknown-NT contact sum exceeds uint64 range")
            unknown_edge_labels.update(map(str, selected_labels[unknown].tolist()))
            known_count = int(np.count_nonzero(known))
            materialized_edges += known_count
            materialized_contact_sum += int(selected_weights[known].astype(np.uint64).sum(dtype=np.uint64))
            if materialized_contact_sum > _UINT64_MAX:
                raise ValueError("materialized contact sum exceeds uint64 range")
            excitatory_edges += int(np.count_nonzero(signs == 1))
            inhibitory_edges += int(np.count_nonzero(signs == -1))
            if known_count:
                known_pre_indices = selected_pre_indices[known]
                row_counts += np.bincount(
                    known_pre_indices, minlength=selected_ids.size
                ).astype(np.uint64, copy=False)

            records = np.empty(selected_count, dtype=_EDGE_DTYPE)
            records["pre"] = selected_pre_indices
            records["post"] = selected_post_indices
            records["weight"] = selected_weights.astype(np.uint16, copy=False)
            records["sign"] = signs
            pending.append(records)
            pending_rows += selected_count
            if pending_rows >= _CHUNK_ROWS:
                pending, pending_rows, path = _flush_chunk(
                    pending, chunk_dir=chunk_dir, chunk_index=chunk_index
                )
                chunk_paths.append(path)
                chunk_index += 1

        if pending:
            pending, pending_rows, path = _flush_chunk(
                pending, chunk_dir=chunk_dir, chunk_index=chunk_index
            )
            chunk_paths.append(path)

        if materialized_edges > _UINT32_MAX:
            raise ValueError("materialized edge count exceeds uint32 limit")
        row_ptr64 = np.empty(selected_ids.size + 1, dtype=np.uint64)
        row_ptr64[0] = 0
        if selected_ids.size:
            np.cumsum(row_counts, dtype=np.uint64, out=row_ptr64[1:])
        if int(row_ptr64[-1]) != materialized_edges:
            raise ValueError("CSR row counts do not match materialized edge count")
        if int(row_ptr64[-1]) > _UINT32_MAX:
            raise ValueError("CSR row pointer exceeds uint32 limit")

        np.save(staging / "neuron_ids.npy", selected_ids.astype(np.uint64, copy=True), allow_pickle=False)
        np.save(staging / "row_ptr.npy", row_ptr64.astype(np.uint32), allow_pickle=False)
        destinations = np.lib.format.open_memmap(
            staging / "destinations.npy",
            mode="w+",
            dtype=np.uint32,
            shape=(materialized_edges,),
        )
        signed_counts = np.lib.format.open_memmap(
            staging / "signed_counts.npy",
            mode="w+",
            dtype=np.int16,
            shape=(materialized_edges,),
        )
        _materialize_chunks(
            chunk_paths,
            expected_materialized_edges=materialized_edges,
            destinations=destinations,
            signed_counts=signed_counts,
        )
        destinations.flush()
        signed_counts.flush()
        del destinations, signed_counts
        for name in ("neuron_ids.npy", "row_ptr.npy", "destinations.npy", "signed_counts.npy"):
            with (staging / name).open("rb") as stream:
                os.fsync(stream.fileno())

        source_files = {
            "annotations": {
                "path": str(annotation),
                "url": source_urls["annotations"],
                "sha256": source_hashes["annotations"],
                "bytes": annotation.stat().st_size,
                "rows": annotation_table.num_rows,
                "schema": _schema_description(annotation_schema),
            },
            "neurotransmitters": {
                "path": str(neurotransmitters),
                "url": source_urls["neurotransmitters"],
                "sha256": source_hashes["neurotransmitters"],
                "bytes": neurotransmitters.stat().st_size,
                "rows": nt_table.num_rows,
                "schema": _schema_description(nt_schema),
            },
            "connectivity": {
                "path": str(connectivity),
                "url": source_urls["connectivity"],
                "sha256": source_hashes["connectivity"],
                "bytes": connectivity.stat().st_size,
                "rows": raw_rows,
                "record_batches": reader.num_record_batches,
                "schema": _schema_description(connectivity_schema),
            },
        }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "materialization": materialization,
            "dataset": "MaleCNS",
            "dataset_version": "v1.0",
            "neuron_count": int(selected_ids.size),
            "edge_count": materialized_edges,
            "contact_sum": materialized_contact_sum,
            "excitatory_edge_count": excitatory_edges,
            "inhibitory_edge_count": inhibitory_edges,
            "counts": {
                "neurons": int(selected_ids.size),
                "edges": materialized_edges,
                "contacts": materialized_contact_sum,
                "excitatory_edges": excitatory_edges,
                "inhibitory_edges": inhibitory_edges,
                "raw_connectivity_rows": raw_rows,
                "raw_connectivity_contact_sum": raw_contact_sum,
                "raw_self_loop_rows": raw_self_loops,
                "selected_pre_endpoint_rows": selected_pre_rows,
                "selected_post_endpoint_rows": selected_post_rows,
                "selected_endpoint_edge_rows": selected_rows,
                "selected_endpoint_contact_sum": selected_contact_sum,
                "omitted_unknown_nt_edges": unknown_edges,
                "omitted_unknown_nt_contact_sum": unknown_contact_sum,
            },
            "selection": {
                "predicate": "annotations.superclass != null",
                "selected_neuron_count": int(selected_ids.size),
                "annotation_row_count": annotation_table.num_rows,
                "status_counts_all": _count_records(annotation_status),
                "status_counts_selected": _count_records(
                    [value for value, selected in zip(annotation_status, selected_mask, strict=True) if selected]
                ),
                "superclass_counts_selected": _count_records(
                    [value for value, selected in zip(annotation_superclasses, selected_mask, strict=True) if selected]
                ),
            },
            "neurotransmitter_policy": {
                "label_column": "consensus_nt",
                "known_signs": _KNOWN_SIGNS,
                "label_aliases": _NT_ALIASES,
                "sign_interpretation": (
                    "Engineering convention for internal CSR edges only; it is not a biological "
                    "certainty and does not infer motor-neuron-to-muscle polarity."
                ),
                "unknown_policy": "retain_node_omit_outgoing_edge",
                "missing_policy": "retain_node_omit_outgoing_edge",
                "confidence_columns": [
                    name
                    for name in (
                        "predicted_nt_confidence",
                        "celltype_predicted_nt_confidence",
                    )
                    if name in nt_schema.names
                ],
                "confidence_threshold": None,
                "confidence_used_for_selection": False,
                "confidence_note": "Published consensus_nt is consumed verbatim; no additional confidence threshold is applied.",
                "all_source_label_counts": _count_records(nt_labels_unsorted),
                "selected_node_label_counts": _count_records(selected_node_labels),
                "omitted_edge_label_counts": _counter_records(unknown_edge_labels),
            },
            "source_sha256": source_hashes,
            "source_files": source_files,
            "provenance": {
                "download_page": _DOWNLOAD_PAGE_URL,
                "source_urls": source_urls,
                "license": "CC BY 4.0",
                "license_url": _LICENSE_URL,
                "license_notice": _LICENSE_NOTICE,
                "modifications": "Filtered annotated nodes and known-NT directed edges into CSR; omitted edges are counted above.",
            },
            "array_sha256": {
                name: _sha256(staging / name)
                for name in (
                    "neuron_ids.npy",
                    "row_ptr.npy",
                    "destinations.npy",
                    "signed_counts.npy",
                )
            },
        }
        _write_json(staging / "manifest.json", manifest)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"output directory already exists: {output}")
        os.replace(staging, output)
        try:
            directory_fd = os.open(output.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return manifest
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


__all__ = ["import_male_cns"]
