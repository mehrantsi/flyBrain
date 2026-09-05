from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import mlx.core as mx
import numpy as np

from flybrain.connectome import PackedConnectome
from flybrain.mlx_engine import MLXEngine, PropagationBackend
from flybrain.parameters import ModelParameters
from flybrain.protocols import RIGHT_SUGAR_GRN_IDS, indices_for_flywire_ids
from flybrain.stimulus import PoissonStimulus


def run_sugar_experiment(
    connectome: PackedConnectome,
    output_path: str | Path,
    *,
    duration_ms: float = 1000.0,
    rate_hz: float = 150.0,
    seed: int = 0,
    propagation: PropagationBackend = "metal",
) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output directory already exists: {output}")

    parameters = ModelParameters()
    steps_float = duration_ms / parameters.dt_ms
    steps = round(steps_float)
    if duration_ms <= 0 or not np.isclose(steps, steps_float, atol=1e-9):
        raise ValueError("duration_ms must be positive and aligned to dt_ms")

    available_ids = {int(flywire_id) for flywire_id in connectome.neuron_ids}
    missing_ids = [
        flywire_id for flywire_id in RIGHT_SUGAR_GRN_IDS if flywire_id not in available_ids
    ]
    targets = indices_for_flywire_ids(
        connectome,
        RIGHT_SUGAR_GRN_IDS,
        allow_missing=True,
    )
    stimulus = PoissonStimulus(
        targets,
        rate_hz,
        dt_ms=parameters.dt_ms,
        seed=seed,
    )
    mx.reset_peak_memory()
    engine = MLXEngine(
        connectome,
        parameters,
        propagation=propagation,
        zero_refractory=targets,
    )
    started = perf_counter()
    spike_counts = engine.run_counts(steps, stimulus)
    mx.synchronize()
    elapsed = perf_counter() - started
    firing_rates = spike_counts.astype(np.float64) / (duration_ms / 1000.0)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "materialization": str(connectome.manifest.get("materialization", "unknown")),
        "duration_ms": duration_ms,
        "steps": steps,
        "dt_ms": parameters.dt_ms,
        "stimulus": "right_sugar_grns",
        "stimulus_rate_hz": rate_hz,
        "stimulus_target_count": int(targets.size),
        "missing_stimulus_ids": missing_ids,
        "seed": seed,
        "propagation": propagation,
        "elapsed_seconds": elapsed,
        "realtime_factor": elapsed / (duration_ms / 1000.0),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "total_spikes": int(spike_counts.sum()),
        "model_parameters": parameters.to_dict(),
        "source_array_sha256": connectome.manifest.get("array_sha256", {}),
    }
    _write_run(output, connectome.neuron_ids, spike_counts, firing_rates, manifest)
    return manifest


def _write_run(
    output_path: str | Path,
    neuron_ids: np.ndarray,
    spike_counts: np.ndarray,
    firing_rates: np.ndarray,
    manifest: dict[str, Any],
) -> None:
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    committed = False
    try:
        np.save(temporary / "neuron_ids.npy", np.asarray(neuron_ids, dtype=np.uint64))
        np.save(temporary / "spike_counts.npy", np.asarray(spike_counts, dtype=np.int32))
        np.save(temporary / "firing_rates_hz.npy", np.asarray(firing_rates, dtype=np.float64))
        manifest["output_array_sha256"] = {
            name: _sha256_file(temporary / name)
            for name in ("neuron_ids.npy", "spike_counts.npy", "firing_rates_hz.npy")
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        committed = True
    finally:
        if not committed and temporary.exists():
            shutil.rmtree(temporary)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["run_sugar_experiment"]
