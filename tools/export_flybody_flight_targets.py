"""Export compact, pair-preserving system-identification targets from FlyBody flight data."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / (
    "work/upstream/flybody-data/datasets_flight-imitation/"
    "flight-dataset_saccade-evasion_augmented.hdf5"
)
DEFAULT_OUTPUT = ROOT / "assets/neuromechfly/flight_system_id_targets_v1.json"
FLYBODY_REPOSITORY = "https://github.com/TuragaLab/flybody"
FLYBODY_COMMIT = "d015e9bfe441bd90ae431bac24c55cb74bdbce26"
SPLIT_MODULUS = 20
SPLIT_RANKS = {
    "train": tuple(range(14)),
    "validation": tuple(range(14, 17)),
    "test": tuple(range(17, 20)),
}
QPOS_REFLECTION_SIGN = (1, -1, 1, -1, 1, -1, 1)
QVEL_REFLECTION_SIGN = (1, -1, 1, -1, 1, -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _number(value: float) -> float:
    value = float(value)
    if abs(value) < 5e-13:
        return 0.0
    return round(value, 9)


def _numbers(values: np.ndarray) -> list[float]:
    return [_number(value) for value in np.asarray(values).reshape(-1)]


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("descriptor values must be finite and non-empty")
    return {
        "min": _number(np.min(values)),
        "max": _number(np.max(values)),
        "mean": _number(np.mean(values)),
        "std": _number(np.std(values)),
    }


def _quaternion_to_euler(quaternions: np.ndarray) -> np.ndarray:
    """Return scalar-first quaternion roll, pitch, yaw in radians."""
    quaternions = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError("trajectory contains a non-unit or invalid quaternion")
    q = quaternions / norms[:, None]
    w, x, y, z = q.T
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw = np.unwrap(yaw)
    return np.column_stack((roll, pitch, yaw))


def _pair_split(pair_rank: int) -> str:
    remainder = pair_rank % SPLIT_MODULUS
    for split, ranks in SPLIT_RANKS.items():
        if remainder in ranks:
            return split
    raise AssertionError(f"no split assigned to pair rank {pair_rank}")


def _descriptor(
    qpos: np.ndarray,
    qvel: np.ndarray,
    timestep_seconds: float,
    trajectory_index: int,
    trajectory_type: str,
    pair_id: str,
    reflection: str,
    split: str,
) -> dict[str, Any]:
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 7:
        raise ValueError(f"trajectory {trajectory_index} has invalid qpos shape {qpos.shape}")
    if qvel.shape != (qpos.shape[0], 6):
        raise ValueError(f"trajectory {trajectory_index} has invalid qvel shape {qvel.shape}")
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
        raise ValueError(f"trajectory {trajectory_index} contains non-finite state")
    if not np.isfinite(timestep_seconds) or timestep_seconds <= 0.0:
        raise ValueError("dataset timestep must be finite and positive")

    relative_xy = qpos[:, :2] - qpos[0, :2]
    planar_displacement = np.linalg.norm(relative_xy, axis=1)
    planar_step = np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=1)
    planar_velocity = qvel[:, :2]
    planar_speed = np.linalg.norm(planar_velocity, axis=1)
    vertical_speed = qvel[:, 2]
    euler = _quaternion_to_euler(qpos[:, 3:])
    w, x, y, z = (qpos[:, 3:] / np.linalg.norm(qpos[:, 3:], axis=1)[:, None]).T
    body_forward = np.column_stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
        )
    )
    body_left = np.column_stack((-body_forward[:, 1], body_forward[:, 0]))
    forward_speed = np.sum(planar_velocity * body_forward, axis=1)
    lateral_speed = np.sum(planar_velocity * body_left, axis=1)
    duration_seconds = max(0.0, (qpos.shape[0] - 1) * timestep_seconds)
    heading_change = euler[-1, 2] - euler[0, 2]

    return {
        "trajectory_index": trajectory_index,
        "trajectory_type": trajectory_type,
        "pair_id": pair_id,
        "reflection": reflection,
        "split": split,
        "sample_count": int(qpos.shape[0]),
        "duration_seconds": _number(duration_seconds),
        "relative_xy_displacement_cm": _numbers(relative_xy[-1]),
        "vertical_displacement_cm": _number(qpos[-1, 2] - qpos[0, 2]),
        "path_length_cm": _number(np.sum(planar_step)),
        "max_relative_xy_radius_cm": _number(np.max(planar_displacement)),
        "planar_speed_cm_s": _stats(planar_speed),
        "forward_speed_cm_s": _stats(forward_speed),
        "lateral_speed_cm_s": _stats(lateral_speed),
        "vertical_speed_cm_s": _stats(vertical_speed),
        "turn_rate_rad_s": _stats(qvel[:, 5]),
        "angular_speed_rad_s": _stats(np.linalg.norm(qvel[:, 3:], axis=1)),
        "heading_change_rad": _number(heading_change),
        "posture_rad": {
            "roll": _stats(euler[:, 0]),
            "pitch": _stats(euler[:, 1]),
            "yaw_unwrapped": _stats(euler[:, 2]),
        },
    }


def _reflection_error(original: np.ndarray, reflected: np.ndarray, signs: tuple[int, ...]) -> float:
    expected = original * np.asarray(signs, dtype=np.float64)
    if expected.shape != reflected.shape:
        raise ValueError("original/reflected trajectory shapes do not match")
    return _number(np.max(np.abs(expected - reflected)))


def build_payload(source: Path) -> dict[str, Any]:
    try:
        import h5py
    except ModuleNotFoundError as error:
        raise SystemExit(
            "h5py is required to export this asset; use the pinned FlyBody environment "
            "at work/upstream/flybody/.venv/bin/python"
        ) from error

    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha256 = sha256_file(source)
    source_relative = str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source)

    with h5py.File(source, "r") as handle:
        timestep_seconds = float(handle["timestep_seconds"][()])
        source_refs = [_text(value) for value in handle["data_source_refs"][:]]
        trajectories_group = handle["trajectories"]
        trajectory_keys = sorted(trajectories_group.keys())
        expected_keys = [f"{index:03d}" for index in range(len(trajectory_keys))]
        if trajectory_keys != expected_keys:
            raise ValueError("trajectory keys must be contiguous zero-padded indices")

        index_groups = {
            family: {
                "original": [int(value) for value in handle["trajectory_type_indices"][f"{family}_original"][:]],
                "reflected": [int(value) for value in handle["trajectory_type_indices"][f"{family}_reflected"][:]],
            }
            for family in ("evasion", "saccade")
        }
        pairs: list[dict[str, Any]] = []
        descriptors: list[dict[str, Any]] = []
        pair_rank = 0
        for family in ("evasion", "saccade"):
            original_indices = index_groups[family]["original"]
            reflected_indices = index_groups[family]["reflected"]
            if len(original_indices) != len(reflected_indices):
                raise ValueError(f"{family} original/reflected groups have different sizes")
            for pair_ordinal, (original_index, reflected_index) in enumerate(
                zip(original_indices, reflected_indices, strict=True)
            ):
                pair_id = f"{family}-{pair_ordinal:03d}"
                split = _pair_split(pair_rank)
                original_group = trajectories_group[f"{original_index:03d}"]
                reflected_group = trajectories_group[f"{reflected_index:03d}"]
                original_qpos = original_group["com_qpos"][:]
                reflected_qpos = reflected_group["com_qpos"][:]
                original_qvel = original_group["com_qvel"][:]
                reflected_qvel = reflected_group["com_qvel"][:]
                if original_qpos.shape != reflected_qpos.shape:
                    raise ValueError(f"pair {pair_id} has different qpos lengths")
                if original_qvel.shape != reflected_qvel.shape:
                    raise ValueError(f"pair {pair_id} has different qvel lengths")
                qpos_error = _reflection_error(
                    original_qpos, reflected_qpos, QPOS_REFLECTION_SIGN
                )
                qvel_error = _reflection_error(
                    original_qvel, reflected_qvel, QVEL_REFLECTION_SIGN
                )
                pair = {
                    "pair_id": pair_id,
                    "family": family,
                    "split": split,
                    "original_index": original_index,
                    "reflected_index": reflected_index,
                    "reflection_max_abs_error": {
                        "com_qpos": qpos_error,
                        "com_qvel": qvel_error,
                    },
                }
                pairs.append(pair)
                original_type = _text(original_group["trajectory_type"][()])
                reflected_type = _text(reflected_group["trajectory_type"][()])
                if original_type != f"{family}_original" or reflected_type != f"{family}_reflected":
                    raise ValueError(f"pair {pair_id} has inconsistent trajectory_type labels")
                descriptors.append(
                    _descriptor(
                        original_qpos,
                        original_qvel,
                        timestep_seconds,
                        original_index,
                        original_type,
                        pair_id,
                        "original",
                        split,
                    )
                )
                descriptors.append(
                    _descriptor(
                        reflected_qpos,
                        reflected_qvel,
                        timestep_seconds,
                        reflected_index,
                        reflected_type,
                        pair_id,
                        "reflected",
                        split,
                    )
                )
                pair_rank += 1

        descriptors.sort(key=lambda item: int(item["trajectory_index"]))
        pairs.sort(key=lambda item: int(item["original_index"]))
        total_samples = sum(int(item["sample_count"]) for item in descriptors)
        pair_counts = {split: sum(item["split"] == split for item in pairs) for split in SPLIT_RANKS}
        trajectory_counts = {
            split: sum(item["split"] == split for item in descriptors) for split in SPLIT_RANKS
        }
        if any(count * 2 != trajectory_counts[split] for split, count in pair_counts.items()):
            raise AssertionError("original/reflected pairs were not kept in one split")

    return {
        "schema": "flybrain.flight-system-id-targets",
        "schema_version": 1,
        "generator": "tools/export_flybody_flight_targets.py",
        "source": {
            "path": source_relative,
            "bytes": source.stat().st_size,
            "sha256": source_sha256,
            "repository": FLYBODY_REPOSITORY,
            "commit": FLYBODY_COMMIT,
            "license": "GPL-3.0-or-later",
            "license_url": "https://www.gnu.org/licenses/gpl-3.0.html",
            "figshare_doi": "10.25378/janelia.25309105.v4",
            "code_license": "Apache-2.0",
            "data_source_refs": source_refs,
        },
        "units": {
            "time": "s",
            "com_position": "cm (x/y only in descriptors; z is relative displacement)",
            "linear_velocity": "cm/s",
            "quaternion": "unitless scalar-first [w, x, y, z]",
            "angular_velocity": "rad/s",
            "posture": "rad, roll/pitch/yaw from scalar-first quaternion",
        },
        "target_contract": {
            "purpose": "compact targets for posture, speed, and turn-dynamics fitting",
            "fit_targets": [
                "relative_xy_displacement_cm",
                "vertical_displacement_cm",
                "planar_speed_cm_s",
                "forward_speed_cm_s",
                "lateral_speed_cm_s",
                "vertical_speed_cm_s",
                "turn_rate_rad_s",
                "posture_rad",
            ],
            "excluded_targets": [
                "absolute room altitude",
                "absolute x/y room position",
                "raw qpos/qvel time series",
            ],
            "reference_frame": "planar displacement is measured from each trajectory's first x/y sample",
        },
        "dataset": {
            "timestep_seconds": _number(timestep_seconds),
            "trajectory_count": len(descriptors),
            "pair_count": len(pairs),
            "total_samples": total_samples,
            "hdf5_layout": {
                "trajectory_group": "trajectories/{index:03d}",
                "position_dataset": "com_qpos (N, 7)",
                "velocity_dataset": "com_qvel (N, 6)",
                "type_dataset": "trajectory_type",
                "pair_index_groups": "trajectory_type_indices/{family}_{original|reflected}",
            },
        },
        "pairing": {
            "families": ["evasion", "saccade"],
            "reflection_transform": {
                "com_qpos_sign": list(QPOS_REFLECTION_SIGN),
                "com_qvel_sign": list(QVEL_REFLECTION_SIGN),
            },
            "pairs": pairs,
        },
        "split": {
            "method": "sorted family/index pair order; pair_rank modulo 20",
            "assignment": {
                "train": "remainder 0..13",
                "validation": "remainder 14..16",
                "test": "remainder 17..19",
            },
            "pair_counts": pair_counts,
            "trajectory_counts": trajectory_counts,
        },
        "trajectories": descriptors,
    }


def write_payload(payload: dict[str, Any], output: Path, force: bool) -> None:
    output = output.resolve()
    if output.exists() and not force:
        raise FileExistsError(f"refusing to replace {output}; pass --force")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, output)


def main() -> None:
    args = parse_args()
    payload = build_payload(args.source)
    write_payload(payload, args.output, args.force)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": sha256_file(args.output.resolve()),
                "trajectory_count": payload["dataset"]["trajectory_count"],
                "pair_count": payload["dataset"]["pair_count"],
                "split": payload["split"]["pair_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
