from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_REQUIRED_COLUMNS = (
    "Presynaptic_ID",
    "Postsynaptic_ID",
    "Presynaptic_Index",
    "Postsynaptic_Index",
    "Connectivity",
    "Excitatory",
    "Excitatory x Connectivity",
)
_ARRAY_NAMES = (
    "neuron_ids.npy",
    "row_ptr.npy",
    "destinations.npy",
    "signed_counts.npy",
)
_UINT32_MAX = int(np.iinfo(np.uint32).max)
_UINT64_MAX = int(np.iinfo(np.uint64).max)
_INT64_MAX = int(np.iinfo(np.int64).max)
_INT64_MIN = int(np.iinfo(np.int64).min)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_integer_token(token: str, *, name: str, row: int) -> int:
    value = token.strip()
    if not value:
        raise ValueError(f"{name} is empty at row {row}")
    try:
        return int(value, 10)
    except ValueError as exc:
        raise ValueError(f"{name} is not an integer at row {row}: {value!r}") from exc


def _load_neuron_ids(path: Path) -> np.ndarray:
    ids: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        first: list[str] | None = None
        for row in reader:
            if row:
                first = row
                break
        if first is None:
            raise ValueError("completeness CSV is empty")

        first_token = first[0].strip()
        try:
            first_id = int(first_token, 10) if first_token else None
        except ValueError:
            first_id = None

        data_rows: list[list[str]] = []
        if first_id is not None:
            data_rows.append(first)
        for row in reader:
            if row:
                data_rows.append(row)

        for row_number, row in enumerate(data_rows, start=1 if first_id is not None else 2):
            if not row or not row[0].strip():
                raise ValueError(f"neuron ID is empty at completeness row {row_number}")
            value = _parse_integer_token(row[0], name="neuron ID", row=row_number)
            if value < 0 or value > _UINT64_MAX:
                raise ValueError(f"neuron ID is outside uint64 range at row {row_number}")
            ids.append(value)

    result = np.asarray(ids, dtype=np.uint64)
    if result.size and np.unique(result).size != result.size:
        raise ValueError("completeness neuron IDs must be unique")
    return result


def _coerce_integer_array(values: Any, *, name: str, unsigned: bool) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    if array.dtype.kind in "iu":
        if unsigned:
            if array.dtype.kind == "i" and array.size and np.any(array < 0):
                raise ValueError(f"{name} contains a negative value")
            if array.size and int(array.max()) > _UINT64_MAX:
                raise ValueError(f"{name} is outside uint64 range")
            return array.astype(np.uint64, copy=False)
        if array.dtype.kind == "u" and array.size and int(array.max()) > _INT64_MAX:
            raise ValueError(f"{name} is outside signed int64 range")
        return array.astype(np.int64, copy=False)

    if array.dtype.kind == "f":
        if array.size and (np.any(~np.isfinite(array)) or np.any(array != np.trunc(array))):
            raise ValueError(f"{name} must contain integer values")
        if unsigned:
            if array.size and (float(array.min()) < 0 or float(array.max()) > _UINT64_MAX):
                raise ValueError(f"{name} is outside uint64 range")
            return array.astype(np.uint64)
        if array.size and (float(array.min()) < _INT64_MIN or float(array.max()) > _INT64_MAX):
            raise ValueError(f"{name} is outside signed int64 range")
        return array.astype(np.int64)

    if array.dtype.kind == "b":
        raise ValueError(f"{name} must contain integer values, not booleans")

    converted: list[int] = []
    for position, item in enumerate(array.tolist()):
        if isinstance(item, bool):
            raise ValueError(  # noqa: TRY004
                f"{name} must contain integer values, not booleans"
            )
        try:
            value = int(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} contains a non-integer value at row {position}") from exc
        if isinstance(item, float) and (not np.isfinite(item) or item != value):
            raise ValueError(f"{name} must contain integer values")
        if not isinstance(item, (str, bytes, int, np.integer)) and not bool(item == value):
            raise ValueError(f"{name} must contain integer values")
        if unsigned:
            if value < 0 or value > _UINT64_MAX:
                raise ValueError(f"{name} is outside uint64 range at row {position}")
        elif value < _INT64_MIN or value > _INT64_MAX:
            raise ValueError(f"{name} is outside signed int64 range at row {position}")
        converted.append(value)

    return np.asarray(converted, dtype=np.uint64 if unsigned else np.int64)


def _load_connectivity(path: Path) -> dict[str, np.ndarray]:
    try:
        from pyarrow import parquet
    except ImportError as exc:  # pragma: no cover - dependency is declared by the package
        raise RuntimeError("pyarrow is required to read connectivity parquet files") from exc

    schema = parquet.read_schema(path)
    missing = [name for name in _REQUIRED_COLUMNS if name not in schema.names]
    if missing:
        raise ValueError(f"connectivity parquet is missing required columns: {', '.join(missing)}")

    table = parquet.read_table(path, columns=list(_REQUIRED_COLUMNS))
    result: dict[str, np.ndarray] = {}
    for name in _REQUIRED_COLUMNS:
        column = table[name]
        if column.null_count:
            raise ValueError(f"connectivity column {name!r} contains null values")
        values = column.to_numpy(zero_copy_only=False)
        result[name] = _coerce_integer_array(
            values,
            name=name,
            unsigned=name.endswith("_ID"),
        )
    return result


def _first_mismatch(actual: np.ndarray, expected: np.ndarray) -> int | None:
    mismatch = np.flatnonzero(actual != expected)
    return int(mismatch[0]) if mismatch.size else None


def _materialize(
    neuron_ids: np.ndarray,
    columns: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    neuron_count = int(neuron_ids.size)
    if neuron_count > _UINT32_MAX:
        raise ValueError("neuron count exceeds uint32 limit")

    pre_id = columns["Presynaptic_ID"]
    post_id = columns["Postsynaptic_ID"]
    pre_index = columns["Presynaptic_Index"]
    post_index = columns["Postsynaptic_Index"]
    connectivity = columns["Connectivity"]
    excitatory = columns["Excitatory"]
    signed = columns["Excitatory x Connectivity"]

    edge_count = int(pre_id.size)
    if any(int(values.size) != edge_count for values in columns.values()):
        raise ValueError("connectivity columns have inconsistent lengths")
    if edge_count > _UINT32_MAX:
        raise ValueError("edge count exceeds uint32 limit")

    for name, index in (("Presynaptic_Index", pre_index), ("Postsynaptic_Index", post_index)):
        invalid = np.flatnonzero((index < 0) | (index >= neuron_count))
        if invalid.size:
            row = int(invalid[0])
            raise ValueError(
                f"{name} is outside neuron table bounds at connectivity row {row}: "
                f"{int(index[row])}"
            )

    expected_pre_id = neuron_ids[pre_index]
    mismatch = _first_mismatch(pre_id, expected_pre_id)
    if mismatch is not None:
        raise ValueError(
            f"Presynaptic_ID does not match Presynaptic_Index at connectivity row {mismatch}"
        )
    expected_post_id = neuron_ids[post_index]
    mismatch = _first_mismatch(post_id, expected_post_id)
    if mismatch is not None:
        raise ValueError(
            f"Postsynaptic_ID does not match Postsynaptic_Index at connectivity row {mismatch}"
        )

    invalid_sign = np.flatnonzero((excitatory != -1) & (excitatory != 1))
    if invalid_sign.size:
        row = int(invalid_sign[0])
        raise ValueError(f"Excitatory must be either -1 or 1 at connectivity row {row}")

    invalid_count = np.flatnonzero(connectivity <= 0)
    if invalid_count.size:
        row = int(invalid_count[0])
        raise ValueError(f"Connectivity must be positive at connectivity row {row}")

    expected_signed = np.where(excitatory == 1, connectivity, -connectivity)
    mismatch = _first_mismatch(signed, expected_signed)
    if mismatch is not None:
        raise ValueError(
            "Excitatory x Connectivity is inconsistent with Excitatory and Connectivity "
            f"at connectivity row {mismatch}"
        )
    outside_int16 = np.flatnonzero((expected_signed < -32768) | (expected_signed > 32767))
    if outside_int16.size:
        row = int(outside_int16[0])
        raise ValueError(f"signed contact count exceeds int16 range at connectivity row {row}")

    if edge_count:
        pair_order = np.lexsort((post_index, pre_index))
        pair_pre = pre_index[pair_order]
        pair_post = post_index[pair_order]
        duplicate = np.flatnonzero(
            (pair_pre[1:] == pair_pre[:-1]) & (pair_post[1:] == pair_post[:-1])
        )
        if duplicate.size:
            pair_position = int(duplicate[0] + 1)
            raise ValueError(
                "duplicate presynaptic/postsynaptic pair "
                f"at connectivity row {int(pair_order[pair_position])}"
            )

    source_order = np.argsort(pre_index, kind="stable")
    sorted_pre = pre_index[source_order]
    sorted_post = post_index[source_order]
    row_counts = np.bincount(sorted_pre, minlength=neuron_count)
    row_ptr64 = np.empty(neuron_count + 1, dtype=np.uint64)
    row_ptr64[0] = 0
    if neuron_count:
        np.cumsum(row_counts, dtype=np.uint64, out=row_ptr64[1:])
    if int(row_ptr64[-1]) > _UINT32_MAX:
        raise ValueError("row pointer exceeds uint32 limit")

    contact_sum = sum(int(value) for value in connectivity)
    if contact_sum > _UINT64_MAX:
        raise ValueError("contact sum exceeds uint64 limit")

    arrays = {
        "neuron_ids.npy": neuron_ids.astype(np.uint64, copy=True),
        "row_ptr.npy": row_ptr64.astype(np.uint32),
        "destinations.npy": sorted_post.astype(np.uint32, copy=True),
        "signed_counts.npy": signed[source_order].astype(np.int16, copy=True),
    }
    counts = {
        "neurons": neuron_count,
        "edges": edge_count,
        "contacts": contact_sum,
        "excitatory_edges": int(np.count_nonzero(signed > 0)),
        "inhibitory_edges": int(np.count_nonzero(signed < 0)),
    }
    return arrays, counts


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def pack_connectome(
    completeness_path: str | Path,
    connectivity_path: str | Path,
    output_path: str | Path,
    materialization: str | int,
) -> dict[str, Any]:
    """Compile published FlyWire CSV and parquet files into a packed CSR directory."""

    completeness = Path(completeness_path)
    connectivity = Path(connectivity_path)
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output directory already exists: {output}")

    completeness_hash = _sha256_file(completeness)
    connectivity_hash = _sha256_file(connectivity)
    neuron_ids = _load_neuron_ids(completeness)
    columns = _load_connectivity(connectivity)
    arrays, counts = _materialize(neuron_ids, columns)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent)))
    committed = False
    try:
        for name, array in arrays.items():
            target = temporary / name
            np.save(target, array, allow_pickle=False)
            with target.open("rb") as stream:
                os.fsync(stream.fileno())

        array_hashes = {name: _sha256_file(temporary / name) for name in _ARRAY_NAMES}
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "materialization": str(materialization),
            "neuron_count": counts["neurons"],
            "edge_count": counts["edges"],
            "contact_sum": counts["contacts"],
            "excitatory_edge_count": counts["excitatory_edges"],
            "inhibitory_edge_count": counts["inhibitory_edges"],
            "counts": counts,
            "source_sha256": {
                "completeness": completeness_hash,
                "connectivity": connectivity_hash,
            },
            "array_sha256": array_hashes,
        }
        _write_manifest(temporary / "manifest.json", manifest)

        if output.exists() or output.is_symlink():
            raise FileExistsError(f"output directory already exists: {output}")
        os.replace(temporary, output)
        committed = True
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
        if not committed and temporary.exists():
            shutil.rmtree(temporary)


__all__ = ["pack_connectome"]
