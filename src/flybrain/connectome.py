from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class PackedConnectome:
    neuron_ids: npt.NDArray[np.uint64]
    row_ptr: npt.NDArray[np.uint32]
    destinations: npt.NDArray[np.uint32]
    signed_counts: npt.NDArray[np.int16]
    manifest: dict[str, Any]
    path: Path | None = None

    @property
    def neuron_count(self) -> int:
        return int(self.neuron_ids.size)

    @property
    def edge_count(self) -> int:
        return int(self.destinations.size)

    @classmethod
    def load(cls, path: str | Path, mmap_mode: str | None = "r") -> PackedConnectome:
        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text())
        connectome = cls(
            neuron_ids=np.load(root / "neuron_ids.npy", mmap_mode=mmap_mode),
            row_ptr=np.load(root / "row_ptr.npy", mmap_mode=mmap_mode),
            destinations=np.load(root / "destinations.npy", mmap_mode=mmap_mode),
            signed_counts=np.load(root / "signed_counts.npy", mmap_mode=mmap_mode),
            manifest=manifest,
            path=root,
        )
        connectome.validate()
        return connectome

    @classmethod
    def from_arrays(
        cls,
        neuron_ids: npt.ArrayLike,
        row_ptr: npt.ArrayLike,
        destinations: npt.ArrayLike,
        signed_counts: npt.ArrayLike,
        manifest: dict[str, Any] | None = None,
    ) -> PackedConnectome:
        connectome = cls(
            neuron_ids=np.asarray(neuron_ids, dtype=np.uint64),
            row_ptr=np.asarray(row_ptr, dtype=np.uint32),
            destinations=np.asarray(destinations, dtype=np.uint32),
            signed_counts=np.asarray(signed_counts, dtype=np.int16),
            manifest={} if manifest is None else manifest,
        )
        connectome.validate()
        return connectome

    def validate(self) -> None:
        if self.neuron_ids.ndim != 1:
            raise ValueError("neuron_ids must be one-dimensional")
        if self.row_ptr.shape != (self.neuron_count + 1,):
            raise ValueError("row_ptr must have neuron_count + 1 entries")
        if self.destinations.ndim != 1 or self.signed_counts.ndim != 1:
            raise ValueError("edge arrays must be one-dimensional")
        if self.destinations.size != self.signed_counts.size:
            raise ValueError("destinations and signed_counts must have equal lengths")
        if int(self.row_ptr[0]) != 0 or int(self.row_ptr[-1]) != self.edge_count:
            raise ValueError("row_ptr endpoints do not match the edge arrays")
        if np.any(self.row_ptr[1:] < self.row_ptr[:-1]):
            raise ValueError("row_ptr must be non-decreasing")
        if self.edge_count and int(self.destinations.max()) >= self.neuron_count:
            raise ValueError("destination index is outside the neuron table")
        if self.edge_count and np.any(self.signed_counts == 0):
            raise ValueError("zero-weight edges must not be stored")
