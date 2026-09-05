from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from flybrain.connectome import PackedConnectome

_ARRAY_DTYPES = {
    "neuron_ids.npy": np.dtype(np.uint64),
    "row_ptr.npy": np.dtype(np.uint32),
    "destinations.npy": np.dtype(np.uint32),
    "signed_counts.npy": np.dtype(np.int16),
}


@dataclass(frozen=True, slots=True)
class PackAudit:
    materialization: str
    neuron_count: int
    edge_count: int
    contact_sum: int
    excitatory_edge_count: int
    inhibitory_edge_count: int
    maximum_out_degree: int
    array_bytes: int
    array_hashes_verified: bool
    source_hashes_verified: bool | None

    def to_dict(self) -> dict[str, str | int | bool | None]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_pack(
    pack_path: str | Path,
    *,
    completeness_path: str | Path | None = None,
    connectivity_path: str | Path | None = None,
) -> PackAudit:
    root = Path(pack_path)
    connectome = PackedConnectome.load(root)
    manifest = connectome.manifest

    arrays = {
        "neuron_ids.npy": connectome.neuron_ids,
        "row_ptr.npy": connectome.row_ptr,
        "destinations.npy": connectome.destinations,
        "signed_counts.npy": connectome.signed_counts,
    }
    for name, expected_dtype in _ARRAY_DTYPES.items():
        if arrays[name].dtype != expected_dtype:
            raise ValueError(f"{name} has dtype {arrays[name].dtype}, expected {expected_dtype}")

    if np.unique(connectome.neuron_ids).size != connectome.neuron_count:
        raise ValueError("packed neuron IDs are not unique")

    contact_sum = int(np.abs(connectome.signed_counts.astype(np.int64)).sum(dtype=np.uint64))
    excitatory = int(np.count_nonzero(connectome.signed_counts > 0))
    inhibitory = int(np.count_nonzero(connectome.signed_counts < 0))
    calculated = {
        "neuron_count": connectome.neuron_count,
        "edge_count": connectome.edge_count,
        "contact_sum": contact_sum,
        "excitatory_edge_count": excitatory,
        "inhibitory_edge_count": inhibitory,
    }
    for key, value in calculated.items():
        if int(manifest.get(key, -1)) != value:
            raise ValueError(f"manifest {key} does not match packed arrays")

    expected_hashes = manifest.get("array_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("manifest does not contain array_sha256")  # noqa: TRY004
    for name in _ARRAY_DTYPES:
        if expected_hashes.get(name) != _sha256(root / name):
            raise ValueError(f"SHA256 mismatch for {name}")

    source_verified: bool | None = None
    if (completeness_path is None) != (connectivity_path is None):
        raise ValueError("both source paths are required to verify source hashes")
    if completeness_path is not None and connectivity_path is not None:
        expected_sources = manifest.get("source_sha256")
        if not isinstance(expected_sources, dict):
            raise ValueError("manifest does not contain source_sha256")
        actual_sources = {
            "completeness": _sha256(Path(completeness_path)),
            "connectivity": _sha256(Path(connectivity_path)),
        }
        if actual_sources != expected_sources:
            raise ValueError("source SHA256 does not match the manifest")
        source_verified = True

    degrees = np.diff(connectome.row_ptr.astype(np.uint64))
    return PackAudit(
        materialization=str(manifest["materialization"]),
        neuron_count=connectome.neuron_count,
        edge_count=connectome.edge_count,
        contact_sum=contact_sum,
        excitatory_edge_count=excitatory,
        inhibitory_edge_count=inhibitory,
        maximum_out_degree=int(degrees.max(initial=0)),
        array_bytes=sum(int(array.nbytes) for array in arrays.values()),
        array_hashes_verified=True,
        source_hashes_verified=source_verified,
    )


__all__ = ["PackAudit", "audit_pack"]
