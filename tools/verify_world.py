from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rust-bin", type=Path, default=Path("target/release/flybrain-world"))
    parser.add_argument("--assets", type=Path, default=Path("assets/neuromechfly"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256_f64(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    rust = json.loads(
        subprocess.check_output(
            [
                str(args.rust_bin),
                "verify",
                "--assets",
                str(args.assets),
                "--steps",
                str(args.steps),
            ],
            text=True,
        )
    )
    manifest = json.loads((args.assets / "manifest.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(args.assets / "fly.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = manifest["neutral_control"]
    mujoco.mj_forward(model, data)
    for _ in range(args.steps):
        mujoco.mj_step(model, data)

    rust_qpos = np.asarray(rust["qpos"], dtype=np.float64)
    rust_qvel = np.asarray(rust["qvel"], dtype=np.float64)
    sensor_addresses = [entry["address"] for entry in manifest["sensors"]]
    python_contacts = np.asarray(
        [data.sensordata[address] for address in sensor_addresses], dtype=np.float64
    )
    rust_contacts = np.asarray(rust["contacts"], dtype=np.float64)
    metrics = {
        "schema": "flybrain-world-parity-v2",
        "steps": args.steps,
        "time_seconds": float(data.time),
        "rust_python_time_abs_error": abs(float(data.time) - rust["time_seconds"]),
        "qpos_max_abs_error": float(np.max(np.abs(data.qpos - rust_qpos))),
        "qvel_max_abs_error": float(np.max(np.abs(data.qvel - rust_qvel))),
        "contact_max_abs_error": float(
            np.max(np.abs(python_contacts - rust_contacts))
        ),
        "python_qpos_sha256_f64le": sha256_f64(data.qpos),
        "rust_qpos_sha256_f64le": rust["qpos_sha256_f64le"],
        "python_qvel_sha256_f64le": sha256_f64(data.qvel),
        "rust_qvel_sha256_f64le": rust["qvel_sha256_f64le"],
    }
    metrics["exact"] = all(
        metrics[key] == 0.0
        for key in (
            "rust_python_time_abs_error",
            "qpos_max_abs_error",
            "qvel_max_abs_error",
            "contact_max_abs_error",
        )
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    if not metrics["exact"]:
        raise RuntimeError("Rust and Python MuJoCo traces differ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
