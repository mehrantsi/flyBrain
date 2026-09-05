from __future__ import annotations

import argparse
import json
from pathlib import Path

MN9_LEFT_ID = "720575940660219265"
FEEDING_ACTUATORS = [
    "fly/c_head-c_rostrum-pitch-feeding-position",
    "fly/c_rostrum-c_haustellum-pitch-feeding-position",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    brain = manifest["brain"]
    environment = manifest["environment"]
    first_taste_ms = float(environment["first_taste_window_ms"])
    first_mn9_ms = float(brain["first_mn9_spike_window_ms"])
    latency_ms = first_mn9_ms - first_taste_ms
    body = manifest["body"]
    feeding_actuators = body.get("feeding_actuators", [])
    checks = {
        "render_schema_v2": manifest["schema"] == "flybrain-render-v2",
        "brain_enabled": brain["enabled"] is True,
        "mn9_id_exact_string": brain["motor_neuron_id"] == MN9_LEFT_ID,
        "sensory_ids_are_exact_strings": all(
            isinstance(neuron_id, str) for neuron_id in brain["sensory_neuron_ids"]
        ),
        "external_events_present": int(brain["external_event_count"]) > 0,
        "mn9_spikes_present": int(brain["mn9_spike_count"]) > 0,
        "taste_precedes_mn9": 0.0 <= latency_ms <= 200.0,
        "rostrum_command_engaged": float(brain["peak_feeding_extension"]) >= 0.5,
        "fly_remains_at_food": float(environment["final_food_distance"])
        <= float(environment["taste_radius"]),
        "feeding_actuator_present": feeding_actuators == FEEDING_ACTUATORS,
    }
    result = {
        "schema": "flybrain-sensorimotor-verification-v1",
        "manifest": str(args.manifest),
        "mn9_response_latency_ms": latency_ms,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("sensorimotor verification failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
