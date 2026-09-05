"""Export the pinned FlyBody flight policy for a dependency-free Rust port."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ARCHIVE_SHA256 = "2d9937c9af2baafad1690c1b318791bde417b4d26dd96d4385ab6723d5d58582"
FIGSHARE_DOI = "10.25378/janelia.25309105.v4"
DATA_LICENSE = "GPL-3.0+"
INPUT_ORDER = (
    ("accelerometer", 3),
    ("actuator_activation", 0),
    ("gyro", 3),
    ("joints_pos", 25),
    ("joints_vel", 25),
    ("ref_displacement", 18),
    ("ref_root_quat", 24),
    ("velocimeter", 3),
    ("world_zaxis", 3),
)
INPUT_SIZE = sum(size for _, size in INPUT_ORDER)
HIDDEN_SIZE = 256
OUTPUT_SIZE = 12
LAYER_NORM_EPSILON = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--saved-model",
        type=Path,
        default=Path("work/upstream/flybody-data/trained-fly-policies/flight"),
    )
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=Path("work/upstream/flybody-data/trained-fly-policies.zip"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("assets/neuromechfly"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _register_legacy_independent_typespec_alias() -> None:
    import tensorflow as tf
    import tensorflow_probability as tfp
    from tensorflow.python.framework import type_spec_registry
    from tensorflow_probability.python.distributions import independent

    legacy_name = (
        "tensorflow_probability.python.distributions.independent."
        "Independent_ACTTypeSpec"
    )
    try:
        type_spec_registry.lookup(legacy_name)
        return
    except ValueError:
        pass

    probe = tfp.distributions.Independent(
        tfp.distributions.Normal(0.0, 1.0), reinterpreted_batch_ndims=0
    )
    current_type_spec = type(probe._type_spec)
    alias = type("Independent_ACTTypeSpec", (current_type_spec,), {})
    alias.__module__ = independent.__name__
    type_spec_registry.register(legacy_name)(alias)


def _load_saved_model(path: Path) -> Any:
    import tensorflow as tf

    _register_legacy_independent_typespec_alias()
    return tf.saved_model.load(str(path))


def _deterministic_inputs() -> np.ndarray:
    zeros = np.zeros(INPUT_SIZE, dtype=np.float32)
    ramp = np.linspace(-0.75, 0.75, INPUT_SIZE, dtype=np.float32)
    pattern = ((np.arange(INPUT_SIZE, dtype=np.int32) * 37) % 101).astype(
        np.float32
    )
    pattern = pattern / np.float32(50.0) - np.float32(1.0)
    return np.stack((zeros, ramp, pattern), axis=0).astype(np.float32, copy=False)


def _model_inputs(flat_inputs: np.ndarray) -> dict[str, np.ndarray]:
    if flat_inputs.ndim != 2 or flat_inputs.shape[1] != INPUT_SIZE:
        raise ValueError(f"flat inputs must have shape (batch, {INPUT_SIZE})")
    batch = flat_inputs.shape[0]
    result: dict[str, np.ndarray] = {}
    offset = 0
    for name, size in INPUT_ORDER:
        values = flat_inputs[:, offset : offset + size]
        if name == "actuator_activation":
            shape = (batch, 0)
        elif name == "ref_displacement":
            shape = (batch, 6, 3)
        elif name == "ref_root_quat":
            shape = (batch, 6, 4)
        else:
            shape = (batch, size)
        result[f"walker/{name}"] = values.reshape(shape)
        offset += size
    if offset != INPUT_SIZE:
        raise AssertionError("input layout did not consume all features")
    return result


def _variable_arrays(model: Any) -> list[np.ndarray]:
    variables = list(model._variables)
    expected = {
        0: ("feedforward_mlp_torso/linear/b:0", (256,)),
        1: ("feedforward_mlp_torso/linear/w:0", (104, 256)),
        2: ("feedforward_mlp_torso/layer_norm/offset:0", (256,)),
        3: ("feedforward_mlp_torso/layer_norm/scale:0", (256,)),
        4: ("feedforward_mlp_torso/mlp/linear_0/b:0", (256,)),
        5: ("feedforward_mlp_torso/mlp/linear_0/w:0", (256, 256)),
        6: ("feedforward_mlp_torso/mlp/linear_1/b:0", (256,)),
        7: ("feedforward_mlp_torso/mlp/linear_1/w:0", (256, 256)),
        8: ("MultivariateNormalDiagHead/linear/b:0", (12,)),
        9: ("MultivariateNormalDiagHead/linear/w:0", (256, 12)),
    }
    if len(variables) != 12:
        raise ValueError(f"expected 12 SavedModel variables, found {len(variables)}")
    for index, (name, shape) in expected.items():
        variable = variables[index]
        if variable.name != name or tuple(variable.shape) != shape:
            raise ValueError(
                f"unexpected SavedModel variable {index}: "
                f"{variable.name} {tuple(variable.shape)}; expected {name} {shape}"
            )
    return [np.asarray(variable.numpy(), dtype=np.float32) for variable in variables]


def _tensor_payload(variables: list[np.ndarray]) -> tuple[bytes, list[dict[str, Any]]]:
    tensors = [
        ("torso_linear_weight", 1, "feedforward_mlp_torso/linear/w:0"),
        ("torso_linear_bias", 0, "feedforward_mlp_torso/linear/b:0"),
        ("layer_norm_offset", 2, "feedforward_mlp_torso/layer_norm/offset:0"),
        ("layer_norm_scale", 3, "feedforward_mlp_torso/layer_norm/scale:0"),
        ("mlp_linear_0_weight", 5, "feedforward_mlp_torso/mlp/linear_0/w:0"),
        ("mlp_linear_0_bias", 4, "feedforward_mlp_torso/mlp/linear_0/b:0"),
        ("mlp_linear_1_weight", 7, "feedforward_mlp_torso/mlp/linear_1/w:0"),
        ("mlp_linear_1_bias", 6, "feedforward_mlp_torso/mlp/linear_1/b:0"),
        ("mean_head_weight", 9, "MultivariateNormalDiagHead/linear/w:0"),
        ("mean_head_bias", 8, "MultivariateNormalDiagHead/linear/b:0"),
    ]
    payload = bytearray()
    manifest_tensors = []
    for name, variable_index, source_variable_name in tensors:
        value = np.asarray(variables[variable_index], dtype="<f4", order="C")
        raw = value.tobytes(order="C")
        offset_f32 = len(payload) // 4
        payload.extend(raw)
        manifest_tensors.append(
            {
                "name": name,
                "source_variable_index": variable_index,
                "source_variable_name": source_variable_name,
                "dtype": "f32le",
                "shape": list(value.shape),
                "offset_f32": offset_f32,
                "count": int(value.size),
            }
        )
    return bytes(payload), manifest_tensors


def _fixture(model: Any) -> dict[str, Any]:
    flat_inputs = _deterministic_inputs()
    distribution = model._module(_model_inputs(flat_inputs))
    expected = np.asarray(distribution.distribution.loc.numpy(), dtype=np.float32)
    if expected.shape != (flat_inputs.shape[0], OUTPUT_SIZE):
        raise ValueError(f"SavedModel mean output has unexpected shape {expected.shape}")
    return {
        "schema": "flybody.flight-policy-fixture",
        "schema_version": 1,
        "input_order": [name for name, _ in INPUT_ORDER],
        "input_sizes": [size for _, size in INPUT_ORDER],
        "inputs_flat_f32": flat_inputs.tolist(),
        "expected_mean_f32": expected.tolist(),
        "tolerance_abs": 2e-5,
        "output": "distribution.mean",
    }


def export_policy(
    saved_model: Path,
    source_archive: Path,
    output_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    saved_model = saved_model.resolve()
    source_archive = source_archive.resolve()
    output_dir = output_dir.resolve()
    if output_dir in {Path(output_dir.anchor), Path.home()}:
        raise ValueError(f"refusing unsafe output directory: {output_dir}")
    if not saved_model.is_dir():
        raise FileNotFoundError(saved_model)
    if not source_archive.is_file():
        raise FileNotFoundError(source_archive)
    archive_sha256 = sha256_file(source_archive)
    if archive_sha256 != ARCHIVE_SHA256:
        raise ValueError(
            f"source archive SHA256 {archive_sha256} does not match pinned {ARCHIVE_SHA256}"
        )

    model = _load_saved_model(saved_model)
    variables = _variable_arrays(model)
    payload, tensors = _tensor_payload(variables)
    weights_name = "flybody_flight_policy_v1.f32le"
    manifest_name = "flybody_flight_policy_v1.json"
    fixture_name = "flybody_flight_policy_fixture_v1.json"
    paths = [output_dir / weights_name, output_dir / manifest_name, output_dir / fixture_name]
    if not force:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise FileExistsError(f"outputs already exist: {existing}; pass --force")

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / weights_name
    weights_path.write_bytes(payload)
    fixture_path = output_dir / fixture_name
    fixture = _fixture(model)
    fixture["source_archive_sha256"] = archive_sha256
    fixture["policy_manifest"] = manifest_name
    fixture_path.write_text(json.dumps(fixture, indent=2) + "\n")

    manifest = {
        "schema": "flybody.flight-policy",
        "schema_version": 1,
        "policy": "flight",
        "input_size": INPUT_SIZE,
        "input_order": [
            {"name": name, "size": size} for name, size in INPUT_ORDER
        ],
        "hidden_size": HIDDEN_SIZE,
        "output_size": OUTPUT_SIZE,
        "layer_norm_epsilon": LAYER_NORM_EPSILON,
        "activations": ["tanh", "elu", "elu"],
        "output_head": "mean",
        "scale_head": "omitted",
        "weights_file": weights_name,
        "weights_sha256": sha256_bytes(payload),
        "weights_f32_count": len(payload) // 4,
        "tensors": tensors,
        "source": {
            "figshare_doi": FIGSHARE_DOI,
            "data_license": DATA_LICENSE,
            "archive_sha256": archive_sha256,
            "archive_relative_path": "work/upstream/flybody-data/trained-fly-policies.zip",
            "saved_model_relative_path": "work/upstream/flybody-data/trained-fly-policies/flight",
            "saved_model_pb_sha256": sha256_file(saved_model / "saved_model.pb"),
            "tensorflow": "2.18.1",
            "tensorflow_probability": "0.25.0",
            "legacy_typespec_alias": "tensorflow_probability.python.distributions.independent.Independent_ACTTypeSpec",
        },
        "fixture_file": fixture_name,
    }
    manifest_path = output_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "manifest": str(manifest_path),
        "weights": str(weights_path),
        "fixture": str(fixture_path),
        "weights_sha256": manifest["weights_sha256"],
        "weights_f32_count": manifest["weights_f32_count"],
        "source_archive_sha256": archive_sha256,
    }


def main() -> int:
    args = parse_args()
    print(json.dumps(export_policy(args.saved_model, args.source_archive, args.output_dir, args.force), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
