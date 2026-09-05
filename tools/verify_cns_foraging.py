"""Check sampled CNS foraging evidence without equating movement with food search."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def feeding_sample(sample: dict) -> bool:
    return (
        sample["flight_mode"] == "GROUNDED"
        and sample["behavior_mode"] == "FEED"
        and sample["foraging_mode"] == "FEED"
        and sample["taste_active"]
        and sample["tasted_resource"] is not None
        and sample["mn9_rate_hz"] > 0.0
        and sample["feeding_extension"] > 0.1
    )


def analyze(intact: dict, disconnected: dict) -> dict:
    for report in (intact, disconnected):
        if report["schema"] != "flybrain.cns-world-check" or not report["samples"]:
            raise ValueError("expected a nonempty CNS world report")
    samples = intact["samples"]
    control = disconnected["samples"]
    descents = [s for s in samples if s["foraging_mode"] == "DESCEND"]
    first_descent = descents[0]["time_seconds"] if descents else None
    touchdowns = [
        s
        for s in samples
        if first_descent is not None
        and s["time_seconds"] > first_descent
        and s["flight_mode"] == "GROUNDED"
        and s["contact_count"] >= 2
    ]
    feeding = [s for s in samples if feeding_sample(s)]
    departure = []
    if feeding:
        last_feed = feeding[-1]
        departure = [
            s
            for s in samples
            if s["time_seconds"] > last_feed["time_seconds"]
            and s["flight_mode"] in ("TAKEOFF", "CRUISE")
            and s["brain_flight_drive"] > 0.0
            and math.dist(s["root_position"], last_feed["root_position"]) > 10.0
        ]
    first_taste = next((s["time_seconds"] for s in samples if s["taste_active"]), math.inf)
    checks = {
        "matched_initial_state": intact["summary"]["initial_state_sha256"]
        == disconnected["summary"]["initial_state_sha256"],
        "intact_whole_cns_connected": intact["brain"]["neurons"] == 166_700
        and intact["brain"]["motor_outputs_connected"]
        and intact["brain"]["landing_output_connected"],
        "high_power_approach": any(
            s["foraging_mode"] == "APPROACH"
            and s["brain_flight_drive"] > 0.8
            and s["flight_mode"] == "CRUISE"
            for s in samples
        ),
        "descent_then_physical_touchdown": bool(touchdowns),
        "food_contact_with_neural_feeding": bool(feeding),
        "flight_departure_after_feeding": bool(departure),
        "no_extension_before_food_contact": all(
            s["feeding_extension"] == 0.0 for s in samples if s["time_seconds"] < first_taste
        ),
        "control_covers_first_landing": first_descent is not None
        and control[-1]["time_seconds"] >= first_descent,
        "landing_control_keeps_cns_and_flight": disconnected["brain"]["neurons"] == 166_700
        and disconnected["brain"]["motor_outputs_connected"]
        and not disconnected["brain"]["landing_output_connected"]
        and disconnected["summary"]["population_spikes"] > 0
        and disconnected["summary"]["motor_output_spikes"] > 0
        and any(s["flight_mode"] == "CRUISE" for s in control),
        "disconnected_landing_readout_removes_descent": all(
            s["brain_landing_drive"] == 0.0 and s["foraging_mode"] != "DESCEND" for s in control
        ),
    }
    return {
        "schema": "flybrain.cns-foraging-verification-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "metrics": {
            "first_descent_seconds": first_descent,
            "first_touchdown_seconds": touchdowns[0]["time_seconds"] if touchdowns else None,
            "feeding_sample_count": len(feeding),
            "first_feeding_seconds": feeding[0]["time_seconds"] if feeding else None,
            "last_feeding_seconds": feeding[-1]["time_seconds"] if feeding else None,
            "departure_seconds": departure[0]["time_seconds"] if departure else None,
        },
        "interpretation": "Engineering integration evidence, not animal behavioral validation. "
        "A failed food-search check must not be replaced with a controlled sugar-contact assay.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intact-report", type=Path, required=True)
    parser.add_argument("--landing-disconnected-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    result = analyze(
        json.loads(args.intact_report.read_text()),
        json.loads(args.landing_disconnected_report.read_text()),
    )
    result["reports"] = {
        "intact": str(args.intact_report),
        "landing_disconnected": str(args.landing_disconnected_report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
