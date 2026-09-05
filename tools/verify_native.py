"""Cross-check native Rust/Metal spike counts against the independent MLX engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from flybrain.connectome import PackedConnectome
from flybrain.mlx_engine import MLXEngine
from flybrain.protocols import RIGHT_SUGAR_GRN_IDS, indices_for_flywire_ids
from flybrain.stimulus import CounterStimulus


def _u32_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<u4").tobytes()).hexdigest()


def verify(
    pack_path: Path,
    binary_path: Path,
    *,
    steps: int,
    rate_hz: float,
    seed: int,
    chunk_steps: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="flybrain-native-verify-") as temporary:
        output = Path(temporary) / "run"
        command = [
            str(binary_path),
            "simulate",
            "--pack",
            str(pack_path),
            "--steps",
            str(steps),
            "--rate-hz",
            str(rate_hz),
            "--seed",
            str(seed),
            "--chunk-steps",
            str(chunk_steps),
            "--output",
            str(output),
        ]
        native = json.loads(
            subprocess.run(command, check=True, capture_output=True, text=True).stdout
        )
        native_counts = np.load(output / "spike_counts.npy")
        native_voltage = np.load(output / "voltage_final_mv.npy")
        native_conductance = np.load(output / "conductance_final_mv.npy")

    connectome = PackedConnectome.load(pack_path)
    targets = indices_for_flywire_ids(
        connectome,
        RIGHT_SUGAR_GRN_IDS,
        allow_missing=True,
    )
    stimulus = CounterStimulus(targets, rate_hz, seed=seed)
    engine = MLXEngine(
        connectome,
        propagation="metal",
        zero_refractory=targets,
    )
    mlx_counts = engine.run_counts(steps, stimulus)
    mx.synchronize()
    mlx_voltage = np.asarray(engine.voltage_mv, dtype=np.float32)
    mlx_conductance = np.asarray(engine.conductance_mv, dtype=np.float32)
    native_digest = _u32_sha256(native_counts)
    mlx_digest = _u32_sha256(mlx_counts)
    exact = native_digest == mlx_digest
    result = {
        "materialization": native["materialization"],
        "steps": steps,
        "biological_ms": steps * 0.1,
        "seed": seed,
        "rate_hz": rate_hz,
        "native_total_spikes": native["total_spikes"],
        "mlx_total_spikes": int(mlx_counts.sum()),
        "native_spike_counts_sha256": native_digest,
        "mlx_spike_counts_sha256": mlx_digest,
        "per_neuron_spike_counts_exact": exact,
        "maximum_voltage_error_mv": float(np.max(np.abs(native_voltage - mlx_voltage))),
        "maximum_conductance_error_mv": float(np.max(np.abs(native_conductance - mlx_conductance))),
        "native_elapsed_seconds": native["elapsed_seconds"],
        "native_realtime_factor": native["realtime_factor"],
        "native_chunk_steps": chunk_steps,
    }
    if not exact:
        raise AssertionError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("target/release/flybrain-rs"),
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--rate-hz", type=float, default=150.0)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--chunk-steps", type=int, default=256)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    result = verify(
        args.pack,
        args.binary,
        steps=args.steps,
        rate_hz=args.rate_hz,
        seed=args.seed,
        chunk_steps=args.chunk_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
